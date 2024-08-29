import numpy as np
import torch
from models import GraphGAN
from tqdm import tqdm
from utils import MolecularMetrics, SparseMolecularDataset


class Trainer:
    def __init__(
        self,
        model: GraphGAN,
        dataset: SparseMolecularDataset,
        lr: float = 0.001,
        batch_size: int = 128,
        num_critic: int = 5,
        penalty_strength: float = 10,
        la: float = 1,
        metrics: list[str] | None = None,
        feature_matching: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        self.model = model.to(self.device)
        self.dataset = dataset
        self.learning_rate = lr
        self.optimizer_generator = torch.optim.Adam(
            self.model.generator.parameters(), lr=self.learning_rate
        )
        self.optimizer_discriminator = torch.optim.Adam(
            self.model.discriminator.parameters(), lr=self.learning_rate
        )
        self.optimizer_rl = torch.optim.Adam(
            self.model.rl.parameters(), lr=self.learning_rate
        )
        self.batch_size = batch_size
        self.la = la
        self.feature_matching = feature_matching
        self.penalty_strength = penalty_strength
        self.metrics = metrics or ["validity", "logp"]
        self.num_critic = num_critic
        self.reward_real = []
        self.reward_fake = []
        self.steps = len(dataset) // batch_size

    def train_generator(
        self, adj_tensor: torch.Tensor, node_tensor: torch.Tensor
    ) -> float:
        embedding = self.model.gaussian_noise(self.batch_size)
        edges_logits, nodes_logits = self.model.generator(embedding)
        edges_hat, nodes_hat = self.model.fake_input([edges_logits, nodes_logits])
        logits_fake, features_fake = self.model.discriminator(edges_hat, nodes_hat)

        if self.feature_matching:
            _, features_real = self.model.discriminator(adj_tensor, node_tensor)
            loss_g = torch.sum(
                (torch.mean(features_real, 0) - torch.mean(features_fake, 0)) ** 2
            )
        else:
            loss_g = -torch.mean(logits_fake)

        if self.la < 1:
            value_logits_fake = self.model.rl(edges_hat, nodes_hat)
            loss_rl = -torch.mean(value_logits_fake)
            alpha = torch.abs((loss_g / loss_rl).detach())
            loss = torch.tensor(0.0)
            if self.la > 0:
                loss += self.la * loss_g
            if self.la < 1:
                loss += (1 - self.la) * alpha * loss_rl
        else:
            loss = loss_g

        self.optimizer_generator.zero_grad()
        loss.backward()
        self.optimizer_generator.step()

        return loss.item()

    def train_discriminator(
        self, adj_tensor: torch.Tensor, node_tensor: torch.Tensor
    ) -> float:
        embedding = self.model.gaussian_noise(self.batch_size)

        edges_logits, nodes_logits = self.model.generator(embedding)
        edges_hat, nodes_hat = self.model.fake_input([edges_logits, nodes_logits])
        edges_softmax, nodes_softmax = self.model.fake_input(
            [edges_logits, nodes_logits], False, False
        )

        logits_real, _ = self.model.discriminator(adj_tensor, node_tensor)
        logits_fake, _ = self.model.discriminator(edges_hat, nodes_hat)

        eps = torch.rand(
            logits_real.shape[0], 1, 1, 1, dtype=logits_real.dtype, device=self.device
        )
        x_int0 = (adj_tensor * eps + edges_softmax * (1 - eps)).detach()
        x_int1 = (
            node_tensor * eps.squeeze(-1) + nodes_softmax * (1 - eps.squeeze(-1))
        ).detach()

        grad_penalty = self.gradient_penalty(self.model.discriminator, x_int0, x_int1)
        # calculate loss
        loss_d = -logits_real + logits_fake
        loss_d = torch.mean(loss_d)
        grad_penalty = torch.mean(grad_penalty)
        cost = loss_d + self.penalty_strength * grad_penalty

        self.optimizer_discriminator.zero_grad()
        cost.backward()
        self.optimizer_discriminator.step()

        return cost.item()

    def train_rl(self, adj_tensor: torch.Tensor, node_tensor: torch.Tensor) -> float:
        embedding = self.model.gaussian_noise(self.batch_size)
        edges_logits, nodes_logits = self.model.generator(embedding)
        edges_hat, nodes_hat = self.model.fake_input([edges_logits, nodes_logits])

        logits_real_rl = self.model.rl(adj_tensor, node_tensor)
        logits_fake_rl = self.model.rl(edges_hat, nodes_hat)

        loss_v = (logits_real_rl - self.reward_real) ** 2 + (
            logits_fake_rl - self.reward_fake
        ) ** 2
        loss = torch.mean(loss_v)

        self.optimizer_rl.zero_grad()
        loss.backward()
        self.optimizer_rl.step()

        return loss.item()

    # @torch.compile
    def gradient_penalty(self, func, *args) -> torch.Tensor:
        """Used in discriminator/RL loss calculations: Improved Training of Wasserstein GANs, https://arxiv.org/abs/1704.00028"""
        inputs = [torch.autograd.Variable(x, requires_grad=True) for x in args]
        outputs = func(*inputs)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        grad_outputs = [torch.ones_like(o) for o in outputs]
        grad0, grad1 = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )

        grad_penalty0 = torch.mean((1 - torch.norm(grad0, dim=-1)) ** 2, dim=(-2, -1))
        grad_penalty1 = torch.mean(
            (1 - torch.norm(grad1, dim=-1)) ** 2, dim=-1, keepdim=True
        )

        grad_penalty = grad_penalty0 + grad_penalty1

        return grad_penalty

    def train(self, num_epochs: int) -> None:
        for epoch in range(num_epochs):
            g_losses = []
            d_losses = []
            rl_losses = []
            desc = "Epoch {}/{} | g_loss (avg): {:.2f} ({:.2f}) - d_loss (avg): {:.2f} ({:.2f})"
            if self.la < 1:
                desc += " - rl_loss (avg): {:.2f} ({:.2f})"
            _range = tqdm(
                range(self.steps),
                desc=desc.format(epoch + 1, num_epochs, 0, 0, 0, 0, 0, 0),
            )
            for _ in _range:
                mols, _, _, a, x, _, _, _, _ = self.dataset.next_train_batch(
                    self.batch_size
                )
                a = torch.as_tensor(a, dtype=torch.int64)
                x = torch.as_tensor(x, dtype=torch.int64)
                # convert to one hot
                adj_tensor = (
                    torch.nn.functional.one_hot(a, num_classes=self.model.edge_dim)
                    .float()
                    .to(self.device)
                )
                node_tensor = (
                    torch.nn.functional.one_hot(x, num_classes=self.model.node_dim)
                    .float()
                    .to(self.device)
                )

                # generate real data reward
                self.reward_real = self.reward(mols)

                # get fake data reward
                n, e = self.model.generate(self.batch_size)
                mols = [
                    self.dataset.matrices2mol(
                        n_.cpu().numpy(), e_.cpu().numpy(), strict=True
                    )
                    for n_, e_ in zip(n, e)
                ]
                self.reward_fake = self.reward(mols)

                for _ in range(self.num_critic):
                    d_loss = self.train_discriminator(adj_tensor, node_tensor)
                    d_losses.append(d_loss)

                g_loss = self.train_generator(adj_tensor, node_tensor)
                g_losses.append(g_loss)

                if self.la < 1:
                    rl_loss = self.train_rl(adj_tensor, node_tensor)
                    rl_losses.append(rl_loss)

                to_print = [
                    epoch + 1,
                    num_epochs,
                    g_loss,
                    np.mean(g_losses),
                    d_loss,
                    np.mean(d_losses),
                ]
                if self.la < 1:
                    to_print.extend([rl_loss, np.mean(rl_losses)])
                _range.set_description(desc.format(*to_print))

    def reward(self, mols):
        rr = 1.0
        for m in self.metrics:
            if m == "np":
                rr *= MolecularMetrics.natural_product_scores(mols, norm=True)
            elif m == "logp":
                rr *= MolecularMetrics.water_octanol_partition_coefficient_scores(
                    mols, norm=True
                )
            elif m == "sas":
                rr *= MolecularMetrics.synthetic_accessibility_score_scores(
                    mols, norm=True
                )
            elif m == "qed":
                rr *= MolecularMetrics.quantitative_estimation_druglikeness_scores(
                    mols, norm=True
                )
            elif m == "novelty":
                rr *= MolecularMetrics.novel_scores(mols, self.dataset)
            elif m == "dc":
                rr *= MolecularMetrics.drugcandidate_scores(mols, self.dataset)
            elif m == "unique":
                rr *= MolecularMetrics.unique_scores(mols)
            elif m == "diversity":
                rr *= MolecularMetrics.diversity_scores(mols, self.dataset)
            elif m == "validity":
                rr *= MolecularMetrics.valid_scores(mols)
            else:
                raise RuntimeError("{} is not defined as a metric".format(m))

        return rr.reshape(-1, 1)
