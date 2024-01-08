import click

@click.group()
def cli():
    """Simple CLI calculator."""
    pass

@click.command()
@click.argument('x', type=float)
@click.argument('y', type=float)
@click.option('--verbose', is_flag=True, help='Enable verbose output.')
def add(x, y, verbose):
    """Adds two numbers."""
    result = x + y
    if verbose:
        click.echo(f"The sum of {x} and {y} is {result}.")
    else:
        click.echo(result)


@click.command()
@click.argument('x', type=float)
@click.argument('y', type=float)
@click.option('--verbose', is_flag=True, help='Enable verbose output.')
def subtract(x, y, verbose):
    """Subtracts two numbers."""
    result = x - y
    if verbose:
        click.echo(f"The difference between {x} and {y} is {result}.")
    else:
        click.echo(result)

@click.command()
@click.argument('x', type=float)
@click.option('--verbose', is_flag=True, help='Enable verbose output.')
def multiplywith10(x, verbose):
    result = x * 10
    if verbose:
        click.echo(f"{x} times 10 is {result}.")
    else:
        click.echo(result)



cli.add_command(add)
cli.add_command(subtract)
cli.add_command(multiplywith10)



if __name__ == '__main__':
    cli()
