# tla — terse linear algebra in C++23

Vectors and matrices for graphics. One header, no dependencies.

## Conventions

Fixed, and everything else follows from them:

- **Column-major storage, column vectors, `M * v`.** `data()` goes straight into
  `glUniformMatrix4fv` with `transpose = GL_FALSE`.
- **Right-handed view space**, `look_at` faces −Z.
- **Depth in `[0, 1]`** — native to Vulkan and D3D; OpenGL needs
  `glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE)`, 4.5+.
- **Angles in radians.**

## Design

- **`Vec<T, N>` is generic, so components are `v.x()`, not `v.x`.** Named members would
  need a specialisation per size, and a union swizzle reads an inactive member. The
  parentheses are the price.
- **A default-constructed matrix is zero.** Identity is `Mat4f::identity()` — glm's
  `mat4(1.0f)` is a known trap.
- **`*` is componentwise on vectors and the matrix product on matrices.** The one place
  the two types disagree.
- **`M[r, c]` is in math order**, `M.col(c)` returns a column. There is no
  single-argument `[]`: that one is what breeds the `M[c][r]` confusion.
- **A broken precondition is an `assert`.** Normalising a zero vector or inverting a
  singular matrix is a bug; `normalize_or` and `inverse_or` take a fallback instead.
- **Storage is contiguous and stays that way**, guarded by `static_assert` — GPU upload
  depends on it.

## Example

```cpp
#include <print>
#include "tla/tla.h"

int main() {
    using namespace tla;

    const auto model = translation(Vec3f{1.f, 0.f, 0.f}) * rotation_z(radians(90.f));
    const auto view = look_at(Vec3f{0.f, 0.f, 5.f}, Vec3f{}, Vec3f{0.f, 1.f, 0.f});
    const auto proj = perspective(radians(60.f), 16.f / 9.f, 0.1f, 100.f);

    const auto clip = proj * view * model * Vec4f{1.f, 0.f, 0.f, 1.f};
    std::println("{}", clip.xyz() / clip.w());   // (0.19485572, 0.34641019, 0.980981)
}
```

## Not here

Quaternions, SIMD, and inverses past 4×4. Graphics never needs the last one; the other
two stay unwritten until something asks for them.

## Build

Header-only — put `include/` on the include path, or:

```cmake
add_subdirectory(terse-tla)
target_link_libraries(app PRIVATE tla::tla)
```
