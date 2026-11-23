// SPDX-License-Identifier: Apache-2.0
// Минимальное MLX-расширение: C++-функция + nanobind-биндинг.
//
// Сборка: см. lab/extensions/demo/CMakeLists.txt и build.py.
// В реальном проекте здесь были бы примитивы/ядра из lab/32/33/36/37.

#include <stdexcept>

#include <mlx/mlx.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

mx::array add_arrays(const mx::array &a, const mx::array &b, mx::StreamOrDevice s = {}) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("add: shapes must match");
    }
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("add: dtypes must match");
    }
    return mx::add(a, b, s);
}

NB_MODULE(_ext, m) {
    m.doc() = "Minimal MLX C++/nanobind extension for the lab.";
    m.def("add", &add_arrays, "a"_a, "b"_a, "stream"_a = nb::none(),
          "Elementwise a + b on the MLX stream.");
}
