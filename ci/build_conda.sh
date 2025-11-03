#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

source rapids-configure-sccache

source rapids-date-string

rapids-print-env

rapids-logger "Begin pixi build"

pixi build

package_path="$(ls ./*.conda)"
echo "package_path=$package_path" >> "$GITHUB_ENV"
