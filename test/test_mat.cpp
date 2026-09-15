#include <concepts>
#include <format>
#include <sstream>

#include <doctest.h>

#include "tla/tla.h"

using tla::Mat2f;
using tla::Mat3f;
using tla::Mat4f;
using tla::Vec2f;
using tla::Vec3f;
using tla::Vec4f;

namespace {
    // rows: [1 2 3; 0 1 4; 5 6 0], determinant 1, so the inverse is exact in float
    constexpr Mat3f g_m{
        Vec3f{1.f, 0.f, 5.f},
        Vec3f{2.f, 1.f, 6.f},
        Vec3f{3.f, 4.f, 0.f},
    };

    constexpr Mat3f g_m_inv{
        Vec3f{-24.f, 20.f, -5.f},
        Vec3f{18.f, -15.f, 4.f},
        Vec3f{5.f, -4.f, 1.f},
    };

    constexpr Mat2f g_ab{Vec2f{1.f, 3.f}, Vec2f{2.f, 4.f}}; // rows [1 2; 3 4]
    constexpr Mat2f g_cd{Vec2f{5.f, 7.f}, Vec2f{6.f, 8.f}}; // rows [5 6; 7 8]
    constexpr Mat3f g_zero;

    constexpr float g_eps = 1e-5f;
}

// ============================================================================
// compile time
// ============================================================================

// everything that is constant-evaluable with an exact result is checked here.
// rotation, look_at and perspective are absent because std::cos, std::sin and
// std::tan become constexpr only in C++26 -- those live in the runtime section

namespace {
    // construction and access
    static_assert(g_zero[0, 0] == 0.f);                              // zero, NOT identity
    static_assert(g_zero == Mat3f{Vec3f{}, Vec3f{}, Vec3f{}});
    static_assert(Mat3f::identity() == Mat3f::diagonal(Vec3f{1.f, 1.f, 1.f}));
    static_assert(Mat3f::diagonal(Vec3f{1.f, 2.f, 3.f})[2, 2] == 3.f);

    // element access is in math order
    static_assert(g_m[0, 0] == 1.f);
    static_assert(g_m[0, 2] == 3.f);
    static_assert(g_m[2, 0] == 5.f);
    static_assert(g_m.row(0) == Vec3f{1.f, 2.f, 3.f});
    static_assert(g_m.row(2) == Vec3f{5.f, 6.f, 0.f});
    static_assert(g_m.col(0) == Vec3f{1.f, 0.f, 5.f});
    static_assert(g_m.col(2) == Vec3f{3.f, 4.f, 0.f});

    // col returns a reference, so it writes through
    static_assert([] {
        Mat3f m = g_m;
        m.col(1) = Vec3f{7.f, 7.f, 7.f};
        return m.row(0);
    }() == Vec3f{1.f, 7.f, 3.f});

    static_assert(g_m.top_left<2, 2>() == Mat2f{Vec2f{1.f, 0.f}, Vec2f{2.f, 1.f}});
    static_assert(g_m.top_left<1, 1>()[0, 0] == 1.f);

    // arithmetic
    static_assert(g_ab + g_ab == g_ab * 2.f);
    static_assert(g_ab - g_ab == Mat2f{});
    static_assert(2.f * g_ab == g_ab * 2.f);
    static_assert((g_ab * 2.f) / 2.f == g_ab);
    static_assert(-g_ab == g_ab * -1.f);

    // * is the matrix product, not componentwise
    static_assert(g_ab * g_cd == Mat2f{Vec2f{19.f, 43.f}, Vec2f{22.f, 50.f}});
    static_assert(g_ab * g_cd != g_cd * g_ab);
    static_assert(g_ab * Mat2f::identity() == g_ab);
    static_assert(Mat2f::identity() * g_ab == g_ab);

    // every compound assignment, back to where it started
    static_assert([] {
        Mat2f m = g_ab;
        m += g_cd;
        m -= g_cd;
        m *= 2.f;
        m /= 2.f;
        m *= Mat2f::identity();
        return m;
    }() == g_ab);

    // matvec is a linear combination of columns
    static_assert(g_m * Vec3f{1.f, 0.f, 0.f} == g_m.col(0));
    static_assert(g_m * Vec3f{0.f, 1.f, 0.f} == g_m.col(1));
    static_assert(g_m * Vec3f{1.f, 1.f, 1.f} == g_m.col(0) + g_m.col(1) + g_m.col(2));
    static_assert(g_m * Vec3f{1.f, 2.f, 3.f} == Vec3f{14.f, 14.f, 17.f});

    // non-square shapes compose
    constexpr tla::Mat<float, 2, 3> g_wide{Vec2f{1.f, 4.f}, Vec2f{2.f, 5.f}, Vec2f{3.f, 6.f}};

    static_assert(std::same_as<decltype(transpose(g_wide)), tla::Mat<float, 3, 2> >);
    static_assert(std::same_as<decltype(g_wide * tla::Mat<float, 3, 4>{}), tla::Mat<float, 2, 4> >);
    static_assert(transpose(g_wide)[2, 0] == g_wide[0, 2]);
    static_assert(transpose(transpose(g_wide)) == g_wide);
    static_assert(g_wide * Vec3f{1.f, 0.f, 0.f} == Vec2f{1.f, 4.f});

    // transpose
    static_assert(transpose(transpose(g_m)) == g_m);
    static_assert(transpose(g_m).row(0) == g_m.col(0));
    static_assert(transpose(Mat3f::identity()) == Mat3f::identity());

    // determinant
    static_assert(tla::determinant(Mat3f::identity()) == 1.f);
    static_assert(tla::determinant(g_m) == 1.f);
    static_assert(tla::determinant(Mat2f{Vec2f{3.f, 4.f}, Vec2f{8.f, 6.f}}) == -14.f);
    static_assert(tla::determinant(tla::scale(Vec3f{2.f, 3.f, 4.f})) == 24.f);
    static_assert(tla::determinant(tla::translation(Vec3f{5.f, 6.f, 7.f})) == 1.f);
    static_assert(tla::determinant(g_zero) == 0.f);

    // inverse. exact here because the determinant is 1
    static_assert(inverse(g_m) == g_m_inv);
    static_assert(g_m * inverse(g_m) == Mat3f::identity());
    static_assert(inverse_or(g_zero, g_m) == g_m); // singular: the fallback
    static_assert(inverse_or(g_m, g_zero) == g_m_inv);

    static_assert(tla::approx_eq(g_m, g_m));

    // normal_matrix survives non-uniform scale, the model matrix does not.
    // n is perpendicular to the tangent t, and has to stay that way
    constexpr Vec3f g_n{0.f, 1.f, 1.f};
    constexpr Vec3f g_t{0.f, 1.f, -1.f};
    constexpr auto g_squash = tla::scale(Vec3f{1.f, 1.f, 4.f});
    constexpr auto g_linear = g_squash.top_left<3, 3>();

    static_assert(dot(g_n, g_t) == 0.f);
    static_assert(tla::normal_matrix(g_squash) == Mat3f::diagonal(Vec3f{1.f, 1.f, 0.25f}));
    static_assert(dot(tla::normal_matrix(g_squash) * g_n, g_linear * g_t) == 0.f);
    static_assert(dot(g_linear * g_n, g_linear * g_t) != 0.f);

    // translation moves points and leaves directions alone
    static_assert(tla::translation(Vec3f{1.f, 2.f, 3.f}) * Vec4f{10.f, 20.f, 30.f, 1.f}
                  == Vec4f{11.f, 22.f, 33.f, 1.f});
    static_assert(tla::translation(Vec3f{1.f, 2.f, 3.f}) * Vec4f{10.f, 20.f, 30.f, 0.f}
                  == Vec4f{10.f, 20.f, 30.f, 0.f});

    static_assert(tla::scale(Vec3f{2.f, 3.f, 4.f}) * Vec4f{1.f, 1.f, 1.f, 1.f} == Vec4f{2.f, 3.f, 4.f, 1.f});
    static_assert(tla::scale(Vec3f{2.f}) * Vec4f{1.f, 1.f, 1.f, 1.f} == Vec4f{2.f, 2.f, 2.f, 1.f});

    // ortho puts depth in [0, 1] without a divide
    constexpr auto g_ortho = tla::ortho(-2.f, 2.f, -1.f, 1.f, 1.f, 100.f);

    static_assert(tla::approx_eq(g_ortho * Vec4f{2.f, 1.f, -1.f, 1.f}, Vec4f{1.f, 1.f, 0.f, 1.f}, g_eps));
    static_assert(tla::approx_eq(g_ortho * Vec4f{-2.f, -1.f, -100.f, 1.f}, Vec4f{-1.f, -1.f, 1.f, 1.f}, g_eps));
    static_assert(tla::approx_eq(g_ortho * Vec4f{0.f, 0.f, -1.f, 1.f}, Vec4f{0.f, 0.f, 0.f, 1.f}, g_eps));

    // viewport maps NDC to pixels and leaves depth alone
    constexpr auto g_vp = tla::viewport(0.f, 0.f, 800.f, 600.f);

    static_assert(g_vp * Vec4f{-1.f, -1.f, 0.f, 1.f} == Vec4f{0.f, 0.f, 0.f, 1.f});
    static_assert(g_vp * Vec4f{1.f, 1.f, 1.f, 1.f} == Vec4f{800.f, 600.f, 1.f, 1.f});
    static_assert(g_vp * Vec4f{0.f, 0.f, 0.5f, 1.f} == Vec4f{400.f, 300.f, 0.5f, 1.f});
    static_assert(tla::viewport(100.f, 50.f, 800.f, 600.f) * Vec4f{-1.f, -1.f, 0.f, 1.f}
                  == Vec4f{100.f, 50.f, 0.f, 1.f});
}

// ============================================================================
// runtime
// ============================================================================

TEST_CASE("mat: storage is column-major and contiguous") {
    // not provable at compile time: stepping from one column into the next
    // leaves the array, and a constant expression may not do that
    const auto t = tla::translation(Vec3f{1.f, 2.f, 3.f});
    const float *p = t.data();

    CHECK(p[0] == 1.f);
    CHECK(p[12] == 1.f); // this is the layout glUniformMatrix4fv expects
    CHECK(p[13] == 2.f); // with transpose = GL_FALSE
    CHECK(p[14] == 3.f);
    CHECK(p[15] == 1.f);
}

TEST_CASE("mat: rotation") {
    constexpr Vec4f x{1.f, 0.f, 0.f, 1.f};
    constexpr Vec4f y{0.f, 1.f, 0.f, 1.f};
    constexpr Vec4f z{0.f, 0.f, 1.f, 1.f};
    const float quarter = tla::radians(90.f);

    SUBCASE("each axis turns the next one into the one after it") {
        CHECK(tla::approx_eq(tla::rotation_z(quarter) * x, y, g_eps));
        CHECK(tla::approx_eq(tla::rotation_x(quarter) * y, z, g_eps));
        CHECK(tla::approx_eq(tla::rotation_y(quarter) * z, x, g_eps));
    }

    SUBCASE("the axis itself is fixed") {
        CHECK(tla::approx_eq(tla::rotation_z(quarter) * z, z, g_eps));
    }

    SUBCASE("Rodrigues agrees with the axis-aligned forms") {
        const float a = tla::radians(37.f);
        CHECK(tla::approx_eq(tla::rotation(Vec3f{1.f, 0.f, 0.f}, a), tla::rotation_x(a), g_eps));
        CHECK(tla::approx_eq(tla::rotation(Vec3f{0.f, 1.f, 0.f}, a), tla::rotation_y(a), g_eps));
        CHECK(tla::approx_eq(tla::rotation(Vec3f{0.f, 0.f, 1.f}, a), tla::rotation_z(a), g_eps));
    }

    SUBCASE("a rotation is orthonormal: the transpose is the inverse") {
        const auto m = tla::rotation(normalize(Vec3f{1.f, 2.f, 3.f}), tla::radians(37.f));
        CHECK(tla::approx_eq(transpose(m) * m, Mat4f::identity(), g_eps));
        CHECK(tla::determinant(m) == doctest::Approx(1.f));
    }
}

TEST_CASE("mat: inverse round-trips on a transform chain") {
    const auto m = tla::translation(Vec3f{1.f, 2.f, 3.f})
                   * tla::rotation(normalize(Vec3f{1.f, 2.f, 3.f}), tla::radians(37.f))
                   * tla::scale(Vec3f{2.f, 0.5f, 3.f});

    CHECK(tla::approx_eq(m * inverse(m), Mat4f::identity(), g_eps));
    CHECK(tla::approx_eq(inverse(m) * m, Mat4f::identity(), g_eps));
}

TEST_CASE("mat: look_at builds a right-handed view facing -Z") {
    constexpr Vec3f eye{0.f, 0.f, 5.f};
    const auto view = tla::look_at(eye, Vec3f{}, Vec3f{0.f, 1.f, 0.f});

    SUBCASE("the eye lands at the origin and the target on -Z") {
        CHECK(tla::approx_eq(view * Vec4f{eye.x(), eye.y(), eye.z(), 1.f}, Vec4f{0.f, 0.f, 0.f, 1.f}, g_eps));
        CHECK(tla::approx_eq(view * Vec4f{0.f, 0.f, 0.f, 1.f}, Vec4f{0.f, 0.f, -5.f, 1.f}, g_eps));
    }

    SUBCASE("world up stays up and world right stays right") {
        CHECK(tla::approx_eq(view * Vec4f{0.f, 1.f, 0.f, 0.f}, Vec4f{0.f, 1.f, 0.f, 0.f}, g_eps));
        CHECK(tla::approx_eq(view * Vec4f{1.f, 0.f, 0.f, 0.f}, Vec4f{1.f, 0.f, 0.f, 0.f}, g_eps));
    }

    SUBCASE("a view matrix is invertible back to world space") {
        CHECK(tla::approx_eq(inverse(view) * Vec4f{0.f, 0.f, 0.f, 1.f},
                             Vec4f{eye.x(), eye.y(), eye.z(), 1.f}, g_eps));
    }
}

TEST_CASE("mat: perspective puts depth in [0, 1]") {
    constexpr float z_near = 1.f;
    constexpr float z_far = 100.f;
    const auto p = tla::perspective(tla::radians(90.f), 2.f, z_near, z_far);

    const auto ndc = [&](const Vec4f &v) {
        const auto clip = p * v;
        return clip.xyz() / clip.w();
    };

    SUBCASE("the near plane is 0 and the far plane is 1") {
        CHECK(tla::approx_eq(ndc(Vec4f{0.f, 0.f, -z_near, 1.f}).z(), 0.f, g_eps));
        CHECK(tla::approx_eq(ndc(Vec4f{0.f, 0.f, -z_far, 1.f}).z(), 1.f, g_eps));
    }

    SUBCASE("the field of view lands on the edges of the cube") {
        // fovy 90 means the half-extent equals the distance; aspect widens x
        CHECK(tla::approx_eq(ndc(Vec4f{0.f, 10.f, -10.f, 1.f}).y(), 1.f, g_eps));
        CHECK(tla::approx_eq(ndc(Vec4f{20.f, 0.f, -10.f, 1.f}).x(), 1.f, g_eps));
        CHECK(tla::approx_eq(ndc(Vec4f{0.f, 0.f, -10.f, 1.f}).xy(), Vec2f{}, g_eps));
    }

    SUBCASE("w carries the view-space depth, which is what makes it a perspective divide") {
        CHECK(tla::approx_eq((p * Vec4f{0.f, 0.f, -7.f, 1.f}).w(), 7.f, g_eps));
    }
}

TEST_CASE("mat: the whole pipeline agrees with itself") {
    const auto model = tla::translation(Vec3f{1.f, 0.f, 0.f}) * tla::rotation_z(tla::radians(90.f));
    const auto view = tla::look_at(Vec3f{0.f, 0.f, 5.f}, Vec3f{}, Vec3f{0.f, 1.f, 0.f});
    const auto proj = tla::perspective(tla::radians(60.f), 16.f / 9.f, 0.1f, 100.f);
    const auto vp = tla::viewport(0.f, 0.f, 1600.f, 900.f);

    constexpr Vec4f local{1.f, 0.f, 0.f, 1.f};

    SUBCASE("model puts the point where the factors say") {
        // rotation_z by 90 takes +x to +y, then translation shifts by +x
        CHECK(tla::approx_eq(model * local, Vec4f{1.f, 1.f, 0.f, 1.f}, g_eps));
    }

    SUBCASE("a composed matrix equals applying the factors one at a time") {
        const auto mvp = proj * view * model;
        CHECK(tla::approx_eq(mvp * local, proj * (view * (model * local)), g_eps));
    }

    SUBCASE("the point lands on screen, in front of the camera") {
        const auto clip = proj * view * model * local;
        REQUIRE(clip.w() > 0.f);

        const auto ndc = clip.xyz() / clip.w();
        CHECK(tla::abs(ndc.xy()) == tla::clamp(tla::abs(ndc.xy()), 0.f, 1.f));
        CHECK(ndc.z() >= 0.f);
        CHECK(ndc.z() <= 1.f);

        const auto screen = vp * ndc.xyzw(1.f);
        CHECK(screen.x() >= 0.f);
        CHECK(screen.x() <= 1600.f);
        CHECK(screen.y() >= 0.f);
        CHECK(screen.y() <= 900.f);
    }
}

TEST_CASE("mat: formatting prints rows, aligned by column width") {
    constexpr Mat2f m{Vec2f{10.f, 3.f}, Vec2f{2.f, 4.f}};

    CHECK(std::format("{}", m) == "[ 10  2 ]\n[  3  4 ]");

    SUBCASE("the stream operator cannot drift from the formatter") {
        std::ostringstream os;
        os << m;
        CHECK(os.str() == std::format("{}", m));
    }
}
