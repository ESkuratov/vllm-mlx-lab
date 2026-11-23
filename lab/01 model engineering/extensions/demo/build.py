# SPDX-License-Identifier: Apache-2.0
"""Собрать demo_ext через MLX CMake build."""

from pathlib import Path

from mlx import extension
from setuptools import Distribution

if __name__ == "__main__":
    distribution = Distribution(
        {
            "name": "demo_ext",
            "ext_modules": [extension.CMakeExtension("demo_ext._ext")],
            "package_data": {"demo_ext": ["*.so", "*.dylib", "*.metallib"]},
        }
    )
    cmd = extension.CMakeBuild(distribution)
    cmd.initialize_options()
    cmd.build_temp = Path("build")
    cmd.build_lib = Path("build") / "lib"
    cmd.inplace = True
    cmd.ensure_finalized()
    cmd.run()
