# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)

import pytest


def mantid_is_available():
    try:
        import mantid  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


with_mantid_only = pytest.mark.skipif(not mantid_is_available(),
                                      reason='Mantid framework is unavailable')