import torch
from torch import nn


class GraphConvolutionLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        node_dim: int,
        edge_dim: int,
        dropout: float,
        activation: nn.Module | None = None,
    ) -> None:
        super(GraphConvolutionLayer, self).__init__()
        self.edge_convolutions = nn.ModuleList(
            [nn.Linear(in_features, out_features) for _ in range(edge_dim - 1)]
        )
        self.self_connection = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(
        self, adj: torch.Tensor, nodes: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        annotations = (
            torch.cat((hidden, nodes), dim=-1) if hidden is not None else nodes
        )
        edge_outputs = torch.stack(
            [conv(annotations) for conv in self.edge_convolutions], dim=1
        )
        edge_outputs = torch.matmul(adj, edge_outputs)
        output = torch.sum(edge_outputs, dim=1) + self.self_connection(annotations)

        if self.activation is not None:
            output = self.activation(output)

        output = self.dropout(output)
        return adj, nodes, output


class MultiGraphConvolutionLayer(nn.Module):
    def __init__(
        self,
        dims: tuple[int, ...],
        node_dim: int,
        edge_dim: int,
        dropout: float,
        activation: nn.Module | None = None,
    ):
        super(MultiGraphConvolutionLayer, self).__init__()
        self.first_convolution = GraphConvolutionLayer(
            node_dim, dims[0], node_dim, edge_dim, dropout, activation
        )
        self.other_convolutions = nn.ModuleList(
            [
                GraphConvolutionLayer(
                    node_dim + dims[i],
                    dims[i + 1],
                    node_dim,
                    edge_dim,
                    dropout,
                    activation,
                )
                for i in range(len(dims) - 1)
            ]
        )

    def forward(self, adj: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        adj, nodes, hidden = self.first_convolution(adj, nodes)
        for conv in self.other_convolutions:
            adj, nodes, hidden = conv(adj, nodes, hidden)
        return hidden


class GraphAggregationLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float,
        activation: nn.Module | None = None,
    ) -> None:
        super(GraphAggregationLayer, self).__init__()
        self.sigmoid_layer = nn.Sequential(
            nn.Linear(in_features, out_features), nn.Sigmoid()
        )
        self.activation_layer = (
            nn.Sequential(nn.Linear(in_features, out_features), activation)
            if activation is not None
            else nn.Linear(in_features, out_features)
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        i = self.sigmoid_layer(x)
        j = self.activation_layer(x)
        output = torch.sum(i * j, dim=1)

        if self.activation is not None:
            output = self.activation(output)

        output = self.dropout(output)
        return output


class MultiDenseLayer(nn.Module):
    def __init__(
        self,
        dims: tuple[int, ...],
        dropout: float,
        activation: nn.Module | None = None,
    ) -> None:
        super(MultiDenseLayer, self).__init__()
        layers = []
        for in_features, out_features in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_features, out_features))
            if activation is not None:
                layers.append(activation)
            layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
