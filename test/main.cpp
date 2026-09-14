#include <print>

#include "tla/tla.h"

int main() {
    tla::Vec2i v1{1, 2}, v2{3, 4};

    std::println("{} {}", v1, v2);

    auto v3 = v1 + v2;
    std::println("{}", v3);

    std::println("{}", v3.swizzle<1, 0>());

    constexpr tla::Vec3f a{1.f, 0.f, 0.f}, b{0.f, 1.f, 0.f};
    std::println("dot {} cross {}", dot(a, b), cross(a, b));
    std::println("len {} norm {}", length(a + b), normalize(a + b));
    std::println("lerp {}", lerp(a, b, 0.25f));

    tla::Vec4f q{1.f, 2.f, 3.f, 4.f};
    std::println("w {} len_sq {}", q.w(), length_sq(q));

    const auto model = tla::translation(tla::Vec3f{1.f, 0.f, 0.f}) * tla::rotation_z(tla::radians(90.f));
    const auto view = look_at(tla::Vec3f{0.f, 0.f, 5.f}, tla::Vec3f{}, tla::Vec3f{0.f, 1.f, 0.f});
    const auto proj = tla::perspective(tla::radians(60.f), 16.f / 9.f, 0.1f, 100.f);

    const auto clip = proj * view * model * tla::Vec4f{1.f, 0.f, 0.f, 1.f};
    std::println("clip {}  ndc {}", clip, clip.xyz() / clip.w());
    std::println("col0 {}  row0 {}", model.col(0), model.row(0));
    std::println("inv*m == I: {}", approx_eq(model * inverse(model), tla::Mat4f::identity(), 1e-5f));
    std::println("proj:\n{}", proj);
}