import torch
from models import GraphGAN
from utils import SparseMolecularDataset, Trainer
import argparse
from rdkit import RDLogger

if __name__ == "__main__":
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True

    dataset = SparseMolecularDataset()
    dataset.load("data/gdb9_9nodes.sparsedataset")

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=9, help="Number of nodes")
    parser.add_argument("--node_dim", type=int, default=5, help="Node dimension")
    parser.add_argument("--edge_dim", type=int, default=5, help="Edge dimension")
    parser.add_argument("--z_dim", type=int, default=32, help="Latent dimension")
    parser.add_argument(
        "--generator_dims", type=int, nargs="+", default=[128, 256, 512], help="Generator dimensions"
    )
    parser.add_argument(
        "--conv_dims",
        type=int,
        nargs="+",
        default=[64, 32],
        help="Convolutional discriminator dimensions",
    )
    parser.add_argument("--auxiliary_dim", type=int, default=128, help="Auxiliary discriminator dimension")
    parser.add_argument("--linear_dims", type=int, nargs="+", default=[128], help="Linear discriminator dimensions")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--batch_discriminator", action="store_true", help="Use batch discriminator")
    parser.add_argument("--num_epochs", type=float, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_critic", type=int, default=5, help="Number of critic updates per generator update")
    parser.add_argument("--penalty_strength", type=float, default=10, help="WGAN Penalty strength")
    parser.add_argument("--la", type=float, default=1, help="Balance between WGAN loss vs RL loss (1 means full WGAN and 0 means full RL)")
    parser.add_argument("--metrics", type=str, nargs="+", default=["validity", "logp"], help="Metrics to use in reward")
    parser.add_argument("--feature_matching", action="store_true", help="Use feature matching")
    parser.add_argument("--enable_warnings", action="store_true", help="Enable RDKit warnings")

    config = parser.parse_args()
    config.generator_dims = tuple(config.generator_dims)
    config.discriminator_dims = (
        tuple(config.conv_dims),
        config.auxiliary_dim,
        tuple(config.linear_dims),
    )
    config.metrics = tuple(config.metrics)

    model = GraphGAN(
        num_nodes=config.num_nodes,
        node_dim=config.node_dim,
        edge_dim=config.edge_dim,
        z_dim=config.z_dim,
        generator_dims=config.generator_dims,
        discriminator_dims=config.discriminator_dims,
        dropout=config.dropout,
        batch_discriminator=config.batch_discriminator,
    )

    trainer = Trainer(
        model=model,
        dataset=dataset,
        lr=config.lr,
        batch_size=config.batch_size,
        num_critic=config.num_critic,
        penalty_strength=config.penalty_strength,
        la=config.la,
        metrics=config.metrics,
        feature_matching=config.feature_matching,
        device=device,
    )
    if not config.enable_warnings:
        RDLogger.DisableLog("rdApp.*")

    trainer.train(config.num_epochs)
