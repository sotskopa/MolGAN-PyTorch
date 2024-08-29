import torch
import torch.nn as nn
from .layers import (
    GraphAggregationLayer,
    MultiDenseLayer,
    MultiGraphConvolutionLayer,
)


class Generator(nn.Module):
    def __init__(
        self,
        dims: tuple[int, ...],
        z_dim: int,
        num_nodes: int,
        node_dim: int,
        edge_dim: int,
        dropout: float,
    ) -> None:
        super(Generator, self).__init__()

        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        layers = []
        for in_features, out_features in zip((z_dim,) + dims[:-1], dims):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout, inplace=True))
        self.layers = nn.Sequential(*layers)

        self.edge_layer = nn.Linear(dims[-1], edge_dim * num_nodes * num_nodes)
        self.node_layer = nn.Linear(dims[-1], node_dim * num_nodes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.layers(x)
        edges_logits = self.edge_layer(output).view(
            -1, self.edge_dim, self.num_nodes, self.num_nodes
        )
        edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2
        edges_logits = self.dropout(edges_logits.permute(0, 2, 3, 1))

        nodes_logits = self.node_layer(output)
        nodes_logits = self.dropout(
            nodes_logits.view(-1, self.num_nodes, self.node_dim)
        )

        return edges_logits, nodes_logits


class Encoder(nn.Module):
    def __init__(
        self,
        graph_convolution_dims: tuple[int, ...],
        auxiliary_dim: int,
        activation: nn.Module | None,
        dropout: float,
        node_dim: int,
        edge_dim: int,
    ) -> None:
        super(Encoder, self).__init__()
        self.auxiliary_dim = auxiliary_dim
        self.mulit_gcn_layer = MultiGraphConvolutionLayer(
            graph_convolution_dims, node_dim, edge_dim, dropout, activation
        )
        self.aggregation_layer = GraphAggregationLayer(
            graph_convolution_dims[-1] + node_dim,
            auxiliary_dim,
            dropout,
            activation,
        )

    def forward(
        self,
        adj: torch.Tensor,
        nodes: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        adj_ = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        output = self.mulit_gcn_layer(adj_, nodes)
        if hidden is not None:
            annotations = torch.cat((output, hidden, nodes), -1)
        else:
            annotations = torch.cat((output, nodes), -1)
        output = self.aggregation_layer(annotations)
        return output


class Discriminator(nn.Module):
    def __init__(
        self,
        dims: tuple[tuple[int, ...], int, tuple[int, ...]],
        node_dim: int = 5,
        edge_dim: int = 5,
        dropout: float = 0.0,
        activation: nn.Module | None = nn.Tanh(),
        batch_discriminator: bool = False,
    ) -> None:
        super(Discriminator, self).__init__()
        self.batch_discriminator = batch_discriminator
        graph_convolution_dims, auxiliary_dim, linear_dims = dims
        self.encoder = Encoder(
            graph_convolution_dims=graph_convolution_dims,
            auxiliary_dim=auxiliary_dim,
            activation=activation,
            dropout=dropout,
            node_dim=node_dim,
            edge_dim=edge_dim,
        )
        self.multi_dense_layer = MultiDenseLayer(
            (auxiliary_dim,) + linear_dims, dropout, activation
        )
        if self.batch_discriminator:
            batch_discriminator_dim = auxiliary_dim // 8
            self.linear_1 = nn.Sequential(
                nn.Linear(auxiliary_dim, batch_discriminator_dim),
                nn.Tanh(),
            )
            self.linear_2 = nn.Sequential(
                nn.Linear(batch_discriminator_dim, batch_discriminator_dim),
                nn.Tanh(),
            )
        self.output_layer = nn.Linear(linear_dims[-1], 1)

    def forward(
        self, adj: torch.Tensor, nodes: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.encoder(adj, nodes, hidden)
        output_penultimate = self.multi_dense_layer(output)

        if self.batch_discriminator:
            output_batch = self.linear_1(output)
            output_batch = torch.mean(output_batch, 0, keepdim=True)
            output_batch = self.linear_2(output_batch)
            output_batch = torch.tile(output_batch, (output.shape[0], 1))
            output_penultimate = torch.cat((output_penultimate, output_batch), -1)

        output = self.output_layer(output_penultimate)
        return output, output_penultimate


class RewardNetwork(nn.Module):
    def __init__(
        self,
        dims: tuple[tuple[int, ...], int, tuple[int, ...]],
        node_dim: int,
        edge_dim: int,
        dropout: float,
        activation: nn.Module | None = nn.Tanh(),
    ) -> None:
        super(RewardNetwork, self).__init__()
        graph_convolution_dims, auxiliary_dim, linear_dims = dims
        self.encoder = Encoder(
            graph_convolution_dims=graph_convolution_dims,
            auxiliary_dim=auxiliary_dim,
            activation=activation,
            dropout=dropout,
            node_dim=node_dim,
            edge_dim=edge_dim,
        )
        self.multi_dense_layer = MultiDenseLayer(
            (auxiliary_dim,) + linear_dims, dropout, activation
        )
        self.output_layer = nn.Sequential(nn.Linear(linear_dims[-1], 1), nn.Sigmoid())

    def forward(
        self, adj: torch.Tensor, nodes: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> torch.Tensor:
        output = self.encoder(adj, nodes, hidden)
        output_penultimate = self.multi_dense_layer(output)
        output = self.output_layer(output_penultimate)
        return output


class GraphGAN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        node_dim: int,
        edge_dim: int,
        z_dim: int = 32,
        generator_dims: tuple[int, ...] = (128, 256, 512),
        discriminator_dims: tuple[tuple[int, ...], int, tuple[int, ...]] = (
            (64, 32),
            128,
            (128,),
        ),
        dropout: float = 0.0,
        batch_discriminator: bool = False,
    ) -> None:
        super(GraphGAN, self).__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.z_dim = z_dim
        self.generator_dims = generator_dims
        self.discriminator_dims = discriminator_dims
        self.dropout = dropout
        self.batch_discriminator = batch_discriminator

        self.generator = Generator(
            dims=generator_dims,
            z_dim=z_dim,
            num_nodes=num_nodes,
            node_dim=node_dim,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.discriminator = Discriminator(
            dims=discriminator_dims,
            node_dim=node_dim,
            edge_dim=edge_dim,
            dropout=dropout,
            batch_discriminator=batch_discriminator,
        )
        self.rl = RewardNetwork(
            dims=discriminator_dims,
            node_dim=node_dim,
            edge_dim=edge_dim,
            dropout=dropout,
        )

    def generate(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate `n` artificial samples."""
        with torch.no_grad():
            embedding = self.gaussian_noise(n)
            edges_logits, nodes_logits = self.generator(embedding)
            _, _, argmax = self.postprocess_logits([nodes_logits, edges_logits])
            nodes, edges = argmax
            nodes, edges = torch.argmax(nodes, axis=-1), torch.argmax(edges, axis=-1)

        return nodes, edges

    def postprocess_logits(
        self,
        inputs: list[torch.Tensor],
        temperature: float = 1.0,
    ) -> list[torch.Tensor]:
        """Processes logits into useful form."""

        def listify(x):
            return x if isinstance(x, (list, tuple)) else [x]

        def delistify(x):
            return x if len(x) > 1 else x[0]

        softmax = [
            torch.nn.functional.softmax(e_logits / temperature, dim=-1)
            for e_logits in listify(inputs)
        ]
        gumbel_softmax = [
            torch.nn.functional.gumbel_softmax(e_logits, tau=temperature, hard=False)
            for e_logits in listify(inputs)
        ]
        gumbel_argmax = [
            torch.nn.functional.gumbel_softmax(e_logits, tau=temperature, hard=True)
            for e_logits in listify(inputs)
        ]

        return [delistify(e) for e in (softmax, gumbel_softmax, gumbel_argmax)]

    # @torch.compile
    def gaussian_noise(self, batch_size) -> torch.Tensor:
        """Generate gaussian noise of given batch size"""
        return torch.randn(batch_size, self.z_dim)

    def fake_input(
        self,
        inputs,
        soft_gumbel_softmax=False,
        hard_gumbel_softmax=False,
        temperature=1.0,
    ):
        """Generate fake inputs."""
        (
            (edges_softmax, nodes_softmax),
            (edges_gumbel_softmax, nodes_gumbel_softmax),
            (edges_gumbel_argmax, nodes_gumbel_argmax),
        ) = self.postprocess_logits(inputs, temperature=temperature)

        if soft_gumbel_softmax:
            edges_hat = edges_gumbel_softmax
            nodes_hat = nodes_gumbel_softmax
        elif hard_gumbel_softmax:
            edges_hat = edges_gumbel_argmax
            nodes_hat = nodes_gumbel_argmax
        else:
            edges_hat = edges_softmax
            nodes_hat = nodes_softmax

        return edges_hat, nodes_hat
