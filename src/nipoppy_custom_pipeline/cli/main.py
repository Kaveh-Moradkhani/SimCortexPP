import typer

app = typer.Typer(help="Nipoppy Custom Pipeline (NCP) CLI")

@app.command()
def hello(name: str = "world"):
    """
    Sanity-check command: confirms the CLI wiring works.
    """
    typer.echo(f"Hello, {name}! NCP CLI is working.")

@app.command()
def check_env():
    """
    Basic environment check: prints Python and nipoppy version if available.
    """
    import sys
    typer.echo(f"Python: {sys.version.split()[0]}")
    try:
        import nipoppy  # noqa: F401
        typer.echo("nipoppy: import OK")
    except Exception as e:
        typer.echo(f"nipoppy: import FAILED -> {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
