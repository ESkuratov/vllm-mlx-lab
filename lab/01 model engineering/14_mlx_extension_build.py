# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 14: сборка MLX C++/nanobind расширения.

(Номер 41, потому что 35 уже занят ``inference/22_verbalizer_classification.py``.)

Идея
----
В 32/33/36/37 мы писали Metal-ядра через ``mx.fast.metal_kernel``:
исходник компилируется в рантайме, отдельная сборка не нужна. В реальном
проекте (tiny-llm) ядра живут в C++/Metal extension:

* CMake собирает C++-библиотеку и metallib;
* nanobind делает Python-модуль;
* ``mlx.extension.CMakeExtension`` вызывает CMake из setuptools;
* Python импортирует ``demo_ext._ext``.

Здесь лежит минимальный, но настоящий пример:

``model/extensions/demo/``
    ``CMakeLists.txt``, ``bindings.cpp``, ``build.py``, ``demo_ext/``

Лаборатория проверяет инструменты и, если всё на месте, собирает модуль и
вызывает ``demo_ext.add``. Если cmake/nanobind/Xcode нет — печатает
чек-лист и завершается без ошибки: это учебный build lab, а не тест
окружения.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/01 model engineering/14_mlx_extension_build.py"

Принудительная сборка:
    python "lab/01 model engineering/14_mlx_extension_build.py" --build
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx

from common import note, section, subsection

HERE = Path(__file__).resolve().parent
DEMO = HERE / "extensions" / "demo"


def check_prereqs() -> dict[str, bool]:
    checks = {
        "cmake": shutil.which("cmake") is not None,
        "nanobind": importlib.util.find_spec("nanobind") is not None,
        "mlx": importlib.util.find_spec("mlx") is not None,
        "mlx.extension": importlib.util.find_spec("mlx.extension") is not None,
    }
    # Metal compiler: xcrun metal или отдельный metal в PATH.
    metal = shutil.which("metal")
    if metal is None:
        try:
            proc = subprocess.run(
                ["xcrun", "--find", "metal"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            metal = proc.stdout.strip() if proc.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            metal = None
    checks["metal compiler (optional)"] = bool(metal)
    return checks


def print_checks(checks: dict[str, bool]) -> None:
    subsection("A. Инструменты сборки")
    for name, ok in checks.items():
        print(f"  {'✓' if ok else '✗'} {name}")


def print_setup_help(checks: dict[str, bool]) -> None:
    subsection("B. Чего не хватает")
    if not checks["cmake"]:
        print("  brew install cmake")
    if not checks["nanobind"]:
        print("  uv pip install --python ./vllm-env/bin/python nanobind")
        print("  # или: ./vllm-env/bin/python -m ensurepip && "
              "./vllm-env/bin/python -m pip install nanobind")
    if not checks["metal compiler (optional)"]:
        print("  (опционально) для .metal-ядер нужен полный Xcode:")
        print("  sudo xcode-select -s /Applications/Xcode.app; xcrun --find metal")
        print("  demo-расширение ниже — C++-only, Metal не обязателен.")
    print()
    print("  Файлы расширения уже готовы:")
    for path in sorted(DEMO.rglob("*")):
        if path.is_file():
            print(f"    {path.relative_to(HERE.parent)}")
    note(
        "Этот шаг — про упаковку C++/Metal в Python-модуль. Сами ядра "
        "показаны в `model/09`–`12` через mx.fast.metal_kernel: они не "
        "требуют CMake и работают на этой машине."
    )


def build_and_test() -> bool:
    subsection("B. Сборка demo_ext")
    build = DEMO / "build.py"
    if not build.is_file():
        print(f"  ✗ не найден {build}")
        return False
    print(f"  запуск: {sys.executable} {build}")
    try:
        proc = subprocess.run(
            [sys.executable, str(build)],
            cwd=DEMO,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"  ✗ сборка не запустилась: {error}")
        return False
    if proc.returncode != 0:
        print("  ✗ сборка упала")
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-25:]
        for line in tail:
            print("    " + line)
        return False

    sys.path.insert(0, str(DEMO))
    try:
        from demo_ext import add  # type: ignore
    except Exception as error:  # pragma: no cover - зависит от сборки
        print(f"  ✗ импорт demo_ext не удался: {error}")
        return False

    a = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
    b = mx.array([10.0, 20.0, 30.0], dtype=mx.float32)
    out = add(a, b)
    mx.eval(out)
    if not bool(mx.allclose(out, a + b)):
        print(f"  ✗ add вернул {out}, ожидалось {a + b}")
        return False
    print("  ✓ demo_ext собран и импортирован")
    print(f"  add([1,2,3], [10,20,30]) = {out.tolist()}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build",
        action="store_true",
        help="собирать, даже если проверка инструментов что-то не нашла",
    )
    args = parser.parse_args()

    section("14. Сборка MLX C++/nanobind расширения")
    print(f"  пример: {DEMO}")
    checks = check_prereqs()
    print_checks(checks)

    required = ("cmake", "nanobind", "mlx", "mlx.extension")
    ready = all(checks[key] for key in required)
    if not ready and not args.build:
        print_setup_help(checks)
        section("Вывод")
        note("Инструменты неполные — сборка пропущена, это ожидаемо.")
        note("Самые важные части: CMakeLists.txt, bindings.cpp, build.py.")
        note("После установки cmake/nanobind/Xcode запустите ещё раз.")
        return

    ok = build_and_test()
    if not ok:
        print_setup_help(checks)
        section("Вывод")
        note("Сборка не удалась; смотрите stderr выше и чек-лист.")
        return

    section("Вывод")
    note("CMake + nanobind + mlx.extension — это 'упаковка' C++/Metal.")
    note("В реальном расширении добавьте mlx_build_metallib и примитивы.")
    note("Наши ядра 32/33/36/37 — тот же Metal, но через рантайм-компиляцию.")


if __name__ == "__main__":
    main()
