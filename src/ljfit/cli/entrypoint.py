"""
Entrypoint for command line interface.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..argparser import parser
from ..energy import get_system_energy
from ..lj import get_system_lj_params


def console_entry_point(argv: Sequence[str] | None = None) -> int:
    # get arguments from command line and parse them
    args = parser().parse_args(argv)

    # calculate energies
    if args.energy:
        get_system_energy(args.system)

    # fit LJ parameters
    if args.fit:
        get_system_lj_params(args.system)

    return 0
