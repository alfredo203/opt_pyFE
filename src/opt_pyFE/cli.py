"""Interfaz de línea de comandos (CLI) para opt_pyFE."""

from typing import List, Optional
import typer
from rich.console import Console
from rich.table import Table

import opt_pyFE
from opt_pyFE import ejecutar_analisis

app = typer.Typer(help="opt_pyFE: Paquete de análisis técnico y optimización de portafolios.")
console = Console()


@app.command()
def version():
    """Muestra la versión instalada de opt_pyFE."""
    console.print(f"[bold green]opt_pyFE versión:[/] {opt_pyFE.__version__}")


@app.command()
def optimize(
    tickers: List[str] = typer.Argument(..., help="Lista de tickers (ej: AAPL MSFT NVDA)"),
    start: str = typer.Option("2023-01-01", "--start", "-s", help="Fecha inicial YYYY-MM-DD"),
    end: str = typer.Option("2024-01-01", "--end", "-e", help="Fecha final YYYY-MM-DD"),
    sims: int = typer.Option(3000, "--sims", "-n", help="Número de simulaciones"),
):
    """Ejecuta una optimización rápida de portafolio por simulación de pesos."""
    console.print(f"[bold cyan]Descargando datos y optimizando para:[/] {', '.join(tickers)}")
    try:
        best_weights, retorno, volatilidad, sharpe = ejecutar_analisis(
            tickers=list(tickers),
            start_date=start,
            end_date=end,
            num_portafolios=sims,
        )
        table = Table(title="Resultados del Portafolio Óptimo")
        table.add_column("Métrica / Ticker", style="bold")
        table.add_column("Valor", justify="right")

        for t, w in zip(tickers, best_weights):
            table.add_row(t, f"{w * 100:.2f}%")
        table.add_section()
        table.add_row("Retorno Esperado", f"{retorno:.4f}")
        table.add_row("Volatilidad Esperada", f"{volatilidad:.4f}")
        table.add_row("Ratio de Sharpe", f"{sharpe:.4f}")
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Error durante la optimización:[/] {e}")


def main():
    app()


if __name__ == "__main__":
    app()
