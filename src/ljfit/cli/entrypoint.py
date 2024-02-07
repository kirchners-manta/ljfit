"""
Entrypoint for command line interface.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..argparser import parser
from ..energy import get_system_energy


def console_entry_point(argv: Sequence[str] | None = None) -> int:
    # get arguments from command line and parse them
    args = parser().parse_args(argv)

    if args.energy:
        get_system_energy(args.system)

    return 0
