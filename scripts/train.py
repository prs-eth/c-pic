import click

@click.command()
@click.option('-s', '--svd_alogrithm', type=str, help='SVD algorithm. E.g., svd, tt, qtt, etc.')
@click.option('-l', '--learning_rate', type=float, help='Learning rate.')
@click.option('-n', '--num_epochs', type=int, help='Number of epochs.')
@click.option('-e', '--eval_every', type=int, help='Eval every.')
def main():
    pass