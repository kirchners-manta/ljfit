"""
Entrypoint for command line interface.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..argparser import parser
from ..energy import get_system_energy
from ..helpers import custom_print
from ..lj import get_system_lj_params


def console_entry_point(argv: Sequence[str] | None = None) -> int:

    # get arguments from command line and parse them
    args = parser().parse_args(argv)

    # info
    custom_print(["LJfit", "=====\n"], 0, args.print)

    # calculate energies
    if args.energy:
        get_system_energy(args.system, args.print)

    # fit LJ parameters
    if args.fit:
        get_system_lj_params(args.system, args.print)

    return 0
