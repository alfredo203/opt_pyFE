"""Módulo de retrocompatibilidad para la CLI."""

from trading_unam.cli import app, main, optimize, version

if __name__ == "__main__":
    main()
