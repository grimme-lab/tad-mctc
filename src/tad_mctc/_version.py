# This file is part of tad-mctc.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Module containing the version string.
"""
from __future__ import annotations

import re

import torch

__all__ = ["__version__", "__tversion__"]


def version_tuple(version_string: str) -> tuple[int, ...]:
    """
    Convert a version string (with possible additional version specifications)
    to a tuple following semantic versioning.

    Parameters
    ----------
    version_string : str
        Version string to convert.

    Returns
    -------
    tuple[int, ...]
        Semantic version number as tuple.
    """
    main_version_part = version_string.split("-")[0].split("+")[0].split("_")[0]

    s = main_version_part.split(".")
    if 3 > len(s):
        raise RuntimeError(
            "Version specification does not seem to follow the semantic "
            f"versioning scheme of MAJOR.MINOR.PATCH ({s})."
        )

    # Remove any trailing characters (for example: 2.8.0a0+git7482eb2)
    s = [re.split("[a-zA-Z]", p)[0] for p in s]

    version_numbers = [int(part) for part in s[:3]]
    return tuple(version_numbers)


__version__ = "0.5.1"
"""Version of tad-mctc in semantic versioning."""

__tversion__ = version_tuple(torch.__version__)
"""Version of PyTorch reduced to semantic versioning."""
