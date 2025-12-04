"""
Legacy wrapper that now delegates to `figures.plot_random_suite_figures`.
"""

from __future__ import annotations

from .figures import plot_random_suite_figures


def main() -> None:
    plot_random_suite_figures()


if __name__ == "__main__":
    main()
