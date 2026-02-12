# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import sys
from numba.cuda.utils import redirect_numba_module

_mod = redirect_numba_module(
    locals(), "numba.core.errors", "numba.cuda.core.cuda_errors"
)

# ---------------------------------------------------------------------------
# Restore the pre-redirect exception class hierarchy  (issue #755)
# ---------------------------------------------------------------------------
#
# Before commit 491f552 ("Always use core errors if numba is present"),
# this module defined its own exception classes as *subclasses* of the
# upstream numba.core.errors classes.  This meant
# ``isinstance(upstream_exc, cuda_NumbaError)`` was False -- the CUDA
# compiler relied on this narrowness to short-circuit template search
# and pipeline retries.  The redirect replaced the subclass hierarchy
# with identity (every name became an alias for the upstream class),
# broadening every isinstance check and causing orders-of-magnitude
# slower compile times for types with many overload candidates (e.g.
# strings).
#
# The fix uses dynamic diamond inheritance: for every upstream
# NumbaError descendant, a local subclass is created that inherits
# from both the local parent and the upstream class.  This means:
#
#   isinstance(cuda_TypingError, core_TypingError) -> True
#       (user except clauses still catch CUDA exceptions)
#   isinstance(core_TypingError, cuda_NumbaError)  -> False
#       (narrow isinstance gate restored for compiler internals)

try:
    import numba.core.errors as _ce

    class NumbaError(_ce.NumbaError):
        pass

    # Build local subclasses for every upstream NumbaError descendant.
    # Process in MRO-depth order so parents are created before children.
    _remap = {_ce.NumbaError: NumbaError}
    for _name, _obj in sorted(
        vars(_ce).items(),
        key=lambda pair: len(getattr(pair[1], "__mro__", ())),
    ):
        if (
            isinstance(_obj, type)
            and issubclass(_obj, _ce.NumbaError)
            and _obj is not _ce.NumbaError
        ):
            _parent = _remap.get(_obj.__bases__[0], NumbaError)
            _local = type(_name, (_parent, _obj), {})
            _remap[_obj] = _local
            setattr(_mod, _name, _local)

    setattr(_mod, "NumbaError", NumbaError)

except ImportError:
    pass

sys.modules[__name__] = _mod
