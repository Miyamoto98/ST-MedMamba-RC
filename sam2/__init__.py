# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_module

if not GlobalHydra.instance().is_initialized():
    initialize_config_module(config_module="sam2", version_base=None)
