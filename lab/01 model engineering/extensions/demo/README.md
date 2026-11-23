# demo — минимальное MLX C++/nanobind расширение

Это учебный пример к `lab/01 model engineering/14_mlx_extension_build.py`: как упаковать
C++-функцию в Python-модуль, который линкуется с MLX.

## Требования

- macOS + Apple Silicon;
- CMake ≥ 3.27;
- `nanobind`;
- MLX с CMake-пакетом;
- Xcode/Metal toolchain, если вы добавляете `.metal`-ядра.

## Сборка

```bash
cd lab/01 model engineering/extensions/demo
python build.py
```

После сборки:

```python
from demo_ext import add
import mlx.core as mx
add(mx.array([1, 2]), mx.array([3, 4]))
```

## Что здесь показано

- `CMakeLists.txt` — `find_package(nanobind)`, `find_package(MLX)`,
  `nanobind_add_module`;
- `bindings.cpp` — C++-функция над `mx::array` и `NB_MODULE`;
- `build.py` — `mlx.extension.CMakeExtension` + `setuptools`.

В настоящем расширении (tiny-llm) в CMake добавляется
`mlx_build_metallib(...)`, а в C++ — примитивы/вызовы Metal-ядер.
Наши Metal-ядра из lab/32/33/36/37 показаны через
`mx.fast.metal_kernel`, чтобы не требовать полной сборки.
