# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)

import timeit
import scippneutron as scn
from typing import Callable, Any

from mantid.simpleapi import Load


def bench_scipp_filter(filter_function: Callable, filename: str = None, setup: Callable = None, repeat_times: int = 100, args: Any = None):
    if (filename is None and setup is None) or (filename is not None and setup is not None):
        raise ValueError("Either setup or filename must be passed, but not both.")

    if filename is not None:
        setup_code = f"""
import scippneutron as scn
from mantid.simpleapi import Load
        
ws = Load("{filename}")
ds = scn.from_mantid(ws)
        """
    elif setup is not None:
        setup_code = setup
    else:
        # No idea how it would get here
        raise ValueError("Either setup or filename must be passed, but not both.")

    if args is not None:
        filter_lambda = f"""
from {filter_function.__module__} import {filter_function.__name__}
{filter_function.__name__}(ds, {args})
"""
    else:
        filter_lambda = f"""
from {filter_function.__module__} import {filter_function.__name__}
{filter_function.__name__}(ds)
        """

    output = timeit.repeat(filter_lambda, setup=setup_code, number=1, repeat=repeat_times)
    return min(output)


def bench_mantid_filter(filter_function: Callable, filename: str = None, setup: str = None, repeat_times: int = 100):
    if (filename is None and setup is None) or (filename is not None and setup is not None):
        raise ValueError("Either setup or filename must be passed, but not both.")

    if filename is not None:
        setup = f"""
import scippneutron as scn
from mantid.simpleapi import Load

ws = Load("{filename}", OutputWorkspace="ws")
"""

    def filter_lambda(): return filter_function(InputWorkspace="ws", OutputWorkspace="abc")

    output = timeit.repeat(filter_lambda, setup=setup, number=1, repeat=repeat_times)
    return min(output)
