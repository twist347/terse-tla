#include <format>
#include <limits>
#include <numbers>
#include <sstream>

#include <doctest.h>

#include "tla/tla.h"

using tla::Vec2f;
using tla::Vec2i;
using tla::Vec3f;
using tla::Vec4f;

namespace {
    constexpr Vec3f g_a{1.f, 2.f, 3.f};
    constexpr Vec3f g_b{4.f, 5.f, 6.f};
    constexpr Vec4f g_q{1.f, 2.f, 3.f, 4.f};
    constexpr float g_eps = std::numeric_limits<float>::epsilon() * 8.f;
}

// ============================================================================
// compile time
// ============================================================================

// everything that is constant-evaluable with an exact result is checked here:
// a static_assert proves the value and the constexpr contract at once, and the
// contract is something no runtime check can establish. what is left for the
// test cases below needs libm, a tolerance or std::format -- none constexpr

namespace {
    // construction and access
    static_assert(Vec3f{} == Vec3f{0.f, 0.f, 0.f});
    static_assert(Vec3f{2.f} == Vec3f{2.f, 2.f, 2.f});
    static_assert(Vec3f{1, 0, 0} == Vec3f{1.f, 0.f, 0.f}); // narrowing is allowed on purpose
    static_assert(g_a.x() == 1.f && g_a.y() == 2.f && g_a.z() == 3.f && g_q.w() == 4.f);
    static_assert(g_a[1] == 2.f);
    static_assert(g_a.get<2>() == 3.f);

    // an accessor names storage, it does not copy
    static_assert([] {
        Vec4f v = g_q;
        v.z() = 9.f;
        return v[2];
    }() == 9.f);

    // contiguity within one vector. the same check on a matrix is impossible
    // here: walking from one column into the next leaves the array, which a
    // constant expression may not do. see the runtime case in test_mat.cpp
    static_assert(*g_a.data() == 1.f);
    static_assert(&g_a[1] == g_a.data() + 1);

    static_assert(g_q.swizzle<1, 0>() == Vec2f{2.f, 1.f});
    static_assert(g_q.swizzle<0, 0, 0>() == Vec3f{1.f, 1.f, 1.f});
    static_assert(g_q.xy() == Vec2f{1.f, 2.f});
    static_assert(g_q.xyz() == g_a);
    static_assert(g_a.xyzw(0.f) == Vec4f{1.f, 2.f, 3.f, 0.f});

    // structure bindings
    static_assert([] {
        const auto [x, y, z] = g_a;
        return x * 100.f + y * 10.f + z;
    }() == 123.f);

    // arithmetic
    static_assert(g_a + g_b == Vec3f{5.f, 7.f, 9.f});
    static_assert(g_b - g_a == Vec3f{3.f, 3.f, 3.f});
    static_assert(g_a * g_b == Vec3f{4.f, 10.f, 18.f});
    static_assert(g_b / Vec3f{2.f} == Vec3f{2.f, 2.5f, 3.f});
    static_assert(g_a * 2.f == Vec3f{2.f, 4.f, 6.f});
    static_assert(2.f * g_a == g_a * 2.f);
    static_assert(g_a / 2.f == Vec3f{0.5f, 1.f, 1.5f});
    static_assert(-g_a == Vec3f{-1.f, -2.f, -3.f});
    static_assert(Vec2i{7, 8} / 3 == Vec2i{2, 2}); // integer division keeps scalar semantics

    // every compound assignment, back to where it started
    static_assert([] {
        Vec3f v = g_a;
        v += g_b;
        v -= g_b;
        v *= 2.f;
        v /= 2.f;
        v *= Vec3f{2.f};
        v /= Vec3f{2.f};
        return v;
    }() == g_a);

    // products
    static_assert(dot(g_a, g_b) == 32.f);
    static_assert(dot(Vec3f{1.f, 0.f, 0.f}, Vec3f{0.f, 1.f, 0.f}) == 0.f);

    // cross is right-handed and anticommutative
    static_assert(cross(Vec3f{1.f, 0.f, 0.f}, Vec3f{0.f, 1.f, 0.f}) == Vec3f{0.f, 0.f, 1.f});
    static_assert(cross(Vec3f{0.f, 1.f, 0.f}, Vec3f{0.f, 0.f, 1.f}) == Vec3f{1.f, 0.f, 0.f});
    static_assert(cross(Vec3f{0.f, 0.f, 1.f}, Vec3f{1.f, 0.f, 0.f}) == Vec3f{0.f, 1.f, 0.f});
    static_assert(cross(Vec3f{0.f, 1.f, 0.f}, Vec3f{1.f, 0.f, 0.f}) == Vec3f{0.f, 0.f, -1.f});
    static_assert(cross(g_a, g_a) == Vec3f{});

    static_assert(length_sq(Vec3f{3.f, 4.f, 0.f}) == 25.f);
    static_assert(distance_sq(Vec3f{1.f, 0.f, 0.f}, Vec3f{4.f, 4.f, 0.f}) == 25.f);

    // reflect, GLSL convention: v is incident, n is the unit normal
    static_assert(reflect(Vec3f{1.f, -1.f, 0.f}, Vec3f{0.f, 1.f, 0.f}) == Vec3f{1.f, 1.f, 0.f});
    static_assert(reflect(Vec3f{0.f, -1.f, 0.f}, Vec3f{0.f, 1.f, 0.f}) == Vec3f{0.f, 1.f, 0.f});
    static_assert(reflect(Vec3f{1.f, 0.f, 0.f}, Vec3f{0.f, 1.f, 0.f}) == Vec3f{1.f, 0.f, 0.f});

    // interpolation and clamping
    static_assert(lerp(g_a, g_b, 0.f) == g_a);
    static_assert(lerp(g_a, g_b, 1.f) == g_b);
    static_assert(lerp(g_a, g_b, 0.5f) == Vec3f{2.5f, 3.5f, 4.5f});
    static_assert(min(g_a, g_b) == g_a);
    static_assert(max(g_a, g_b) == g_b);
    static_assert(min(Vec3f{1.f, -5.f, 3.f}, Vec3f{-2.f, 4.f, 3.f}) == Vec3f{-2.f, -5.f, 3.f});
    static_assert(max(Vec3f{1.f, -5.f, 3.f}, Vec3f{-2.f, 4.f, 3.f}) == Vec3f{1.f, 4.f, 3.f});
    static_assert(abs(-g_a) == g_a);
    static_assert(clamp(Vec3f{-1.f, 0.5f, 7.f}, 0.f, 1.f) == Vec3f{0.f, 0.5f, 1.f});
    static_assert(clamp(Vec3f{-1.f, 0.5f, 7.f}, Vec3f{0.f}, Vec3f{2.f, 0.25f, 2.f}) == Vec3f{0.f, 0.25f, 2.f});
    static_assert(saturate(Vec3f{-1.f, 0.5f, 2.f}) == Vec3f{0.f, 0.5f, 1.f});

    // approx_eq mixes absolute and relative tolerance
    static_assert(tla::approx_eq(0.f, g_eps / 2.f));  // near zero the tolerance is absolute
    static_assert(!tla::approx_eq(0.f, 1e-5f));
    static_assert(tla::approx_eq(1e7f, 1e7f + 1.f));  // one ulp at 1e7 is 1.0, and an
    static_assert(!tla::approx_eq(1e7f, 1e7f + 1e3f)); // absolute epsilon would reject it
    static_assert(tla::approx_eq(g_a, Vec3f{1.f, 2.f, 3.f + g_eps}));
    static_assert(!tla::approx_eq(g_a, Vec3f{1.f, 2.f, 3.1f}));

    // angles
    static_assert(tla::approx_eq(tla::radians(180.f), std::numbers::pi_v<float>));
    static_assert(tla::approx_eq(tla::degrees(std::numbers::pi_v<float>), 180.f));
    static_assert(tla::approx_eq(tla::degrees(tla::radians(37.f)), 37.f));
}

// ============================================================================
// runtime
// ============================================================================

TEST_CASE("vec: length and normalization") {
    // std::sqrt is not constexpr before C++26, so this whole group is runtime
    constexpr Vec3f v{3.f, 4.f, 0.f};

    CHECK(length(v) == 5.f);
    CHECK(normalize(v) == Vec3f{0.6f, 0.8f, 0.f});
    CHECK(length(normalize(Vec3f{1.f, 1.f, 1.f})) == doctest::Approx(1.f));
    CHECK(distance(Vec3f{1.f, 0.f, 0.f}, Vec3f{4.f, 4.f, 0.f}) == 5.f);

    SUBCASE("normalize_or falls back below the smallest normal") {
        constexpr Vec3f fallback{0.f, 0.f, 1.f};
        CHECK(normalize_or(Vec3f{}, fallback) == fallback);
        CHECK(normalize_or(Vec3f{std::numeric_limits<float>::denorm_min()}, fallback) == fallback);
        CHECK(normalize_or(v, fallback) == normalize(v));
    }
}

TEST_CASE("vec: formatting") {
    CHECK(std::format("{}", Vec2i{1, 2}) == "(1, 2)");
    CHECK(std::format("{}", Vec3f{1.f, 2.5f, 3.f}) == "(1, 2.5, 3)");

    SUBCASE("the stream operator cannot drift from the formatter") {
        std::ostringstream os;
        os << g_a;
        CHECK(os.str() == std::format("{}", g_a));
    }
}
