#pragma once

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <format>
#include <string>
#include <type_traits>
#include <cassert>
#include <limits>
#include <numbers>
#include <ostream>

// ============================================================================
// concepts
// ============================================================================

namespace tla::detail {
    template<typename T>
    concept Character = std::same_as<T, char> || std::same_as<T, wchar_t>
                        || std::same_as<T, char8_t> || std::same_as<T, char16_t>
                        || std::same_as<T, char32_t>;

    template<typename T>
    concept Plain = std::same_as<T, std::remove_cv_t<T> >;
}

namespace tla {
    template<typename T>
    concept Integral = std::integral<T> && !std::same_as<T, bool>
                       && !detail::Character<T> && detail::Plain<T>;

    template<typename T>
    concept Floating = std::floating_point<T> && detail::Plain<T>;

    template<typename T>
    concept Number = Integral<T> || Floating<T>;
}

// ============================================================================
// vec
// ============================================================================

namespace tla {
    template<Number T, std::size_t N>
        requires (N > 0)
    class Vec {
        std::array<T, N> m_data{};

    public:
        constexpr Vec() noexcept = default;

        template<typename... Args>
            requires (sizeof...(Args) == N && (std::convertible_to<Args, T> && ...))
        constexpr Vec(Args... args) noexcept : m_data{static_cast<T>(args)...} {
        }

        // broadcast: Vec2f{3.f} → {3.f, 3.f}. disabled for N == 1 to avoid overlap with variadic.
        explicit constexpr Vec(T v) noexcept requires (N > 1) {
            m_data.fill(v);
        }

        [[nodiscard]] constexpr auto x(this auto &&self) noexcept -> auto & {
            return self.m_data[0];
        }

        [[nodiscard]] constexpr auto y(this auto &&self) noexcept -> auto & requires (N > 1) {
            return self.m_data[1];
        }

        [[nodiscard]] constexpr auto z(this auto &&self) noexcept -> auto & requires (N > 2) {
            return self.m_data[2];
        }

        [[nodiscard]] constexpr auto w(this auto &&self) noexcept -> auto & requires (N > 3) {
            return self.m_data[3];
        }

        [[nodiscard]] constexpr auto operator[](this auto &&self, std::size_t i) noexcept -> auto & {
            return self.m_data[i];
        }

        template<std::size_t... Is>
            requires (sizeof...(Is) > 0 && ((Is < N) && ...))
        [[nodiscard]] constexpr auto swizzle() const noexcept -> Vec<T, sizeof...(Is)> {
            return Vec<T, sizeof...(Is)>{m_data[Is]...};
        }

        [[nodiscard]] constexpr auto xy() const noexcept -> Vec<T, 2> requires (N >= 2) {
            return swizzle<0, 1>();
        }

        [[nodiscard]] constexpr auto xyz() const noexcept -> Vec<T, 3> requires (N >= 3) {
            return swizzle<0, 1, 2>();
        }

        // extend to homogeneous coordinates
        [[nodiscard]] constexpr auto xyzw(T w) const noexcept -> Vec<T, 4> requires (N == 3) {
            return {m_data[0], m_data[1], m_data[2], w};
        }

        constexpr auto operator+=(const Vec &v) noexcept -> Vec & {
            for (std::size_t i = 0; i < N; ++i) {
                m_data[i] += v.m_data[i];
            }
            return *this;
        }

        constexpr auto operator-=(const Vec &v) noexcept -> Vec & {
            for (std::size_t i = 0; i < N; ++i) {
                m_data[i] -= v.m_data[i];
            }
            return *this;
        }

        constexpr auto operator*=(const Vec &v) noexcept -> Vec & {
            for (std::size_t i = 0; i < N; ++i) {
                m_data[i] *= v.m_data[i];
            }
            return *this;
        }

        constexpr auto operator/=(const Vec &v) noexcept -> Vec & {
            for (std::size_t i = 0; i < N; ++i) {
                m_data[i] /= v.m_data[i];
            }
            return *this;
        }

        constexpr auto operator*=(T s) noexcept -> Vec & {
            for (std::size_t i = 0; i < N; ++i) {
                m_data[i] *= s;
            }
            return *this;
        }

        constexpr auto operator/=(T s) noexcept -> Vec & {
            for (std::size_t i = 0; i < N; ++i) {
                m_data[i] /= s;
            }
            return *this;
        }

        [[nodiscard]] friend constexpr auto operator+(Vec lhs, const Vec &rhs) noexcept -> Vec {
            lhs += rhs;
            return lhs;
        }

        [[nodiscard]] friend constexpr auto operator-(Vec lhs, const Vec &rhs) noexcept -> Vec {
            lhs -= rhs;
            return lhs;
        }

        [[nodiscard]] friend constexpr auto operator*(Vec lhs, const Vec &rhs) noexcept -> Vec {
            lhs *= rhs;
            return lhs;
        }

        [[nodiscard]] friend constexpr auto operator/(Vec lhs, const Vec &rhs) noexcept -> Vec {
            lhs /= rhs;
            return lhs;
        }

        [[nodiscard]] friend constexpr auto operator*(Vec lhs, T rhs) noexcept -> Vec {
            lhs *= rhs;
            return lhs;
        }

        [[nodiscard]] friend constexpr auto operator*(T lhs, Vec rhs) noexcept -> Vec {
            rhs *= lhs;
            return rhs;
        }

        [[nodiscard]] friend constexpr auto operator/(Vec lhs, T rhs) noexcept -> Vec {
            lhs /= rhs;
            return lhs;
        }

        [[nodiscard]] constexpr auto operator-() const noexcept -> Vec {
            Vec res;
            for (std::size_t i = 0; i < N; ++i) {
                res.m_data[i] = -m_data[i];
            }
            return res;
        }

        [[nodiscard]] constexpr auto operator==(const Vec &) const noexcept -> bool = default;

        // contiguity is a hard invariant: GPU upload relies on it
        [[nodiscard]] constexpr auto data(this auto &&self) noexcept -> auto * {
            static_assert(sizeof(Vec) == sizeof(T) * N, "padded vector: storage is not contiguous");
            return self.m_data.data();
        }

        // for structure binding
        template<std::size_t I>
        [[nodiscard]] constexpr auto get(this auto &&self) noexcept -> auto & {
            return self.m_data[I];
        }
    };

    template<Number T, std::size_t N>
    [[nodiscard]] constexpr auto dot(const Vec<T, N> &a, const Vec<T, N> &b) noexcept -> T {
        T res{};
        for (std::size_t i = 0; i < N; ++i) {
            res += a[i] * b[i];
        }
        return res;
    }

    template<Number T>
    [[nodiscard]] constexpr auto cross(const Vec<T, 3> &a, const Vec<T, 3> &b) noexcept -> Vec<T, 3> {
        return {
            a.y() * b.z() - a.z() * b.y(),
            a.z() * b.x() - a.x() * b.z(),
            a.x() * b.y() - a.y() * b.x(),
        };
    }

    template<Number T, std::size_t N>
    [[nodiscard]] constexpr auto length_sq(const Vec<T, N> &v) noexcept -> T {
        return dot(v, v);
    }

    // not constexpr: std::sqrt becomes constexpr only in C++26.
    template<Floating T, std::size_t N>
    [[nodiscard]] auto length(const Vec<T, N> &v) noexcept -> T {
        return std::sqrt(length_sq(v));
    }

    // the threshold is the smallest normal: below it the length is denormal and the division is garbage
    template<Floating T, std::size_t N>
    [[nodiscard]] auto normalize(const Vec<T, N> &v) noexcept -> Vec<T, N> {
        const auto l = length(v);
        assert(l >= std::numeric_limits<T>::min() && "normalize of a zero-length vector");
        return v / l;
    }

    template<Floating T, std::size_t N>
    [[nodiscard]] auto normalize_or(const Vec<T, N> &v, const Vec<T, N> &fallback) noexcept -> Vec<T, N> {
        const auto l = length(v);
        return l >= std::numeric_limits<T>::min() ? v / l : fallback;
    }

    template<Number T, std::size_t N>
    [[nodiscard]] constexpr auto distance_sq(const Vec<T, N> &a, const Vec<T, N> &b) noexcept -> T {
        return length_sq(a - b);
    }

    template<Floating T, std::size_t N>
    [[nodiscard]] auto distance(const Vec<T, N> &a, const Vec<T, N> &b) noexcept -> T {
        return length(a - b);
    }

    // GLSL convention: v is the incident vector, n is normalized
    template<Floating T, std::size_t N>
    [[nodiscard]] constexpr auto reflect(const Vec<T, N> &v, const Vec<T, N> &n) noexcept -> Vec<T, N> {
        return v - n * (T{2} * dot(v, n));
    }

    // Floating only: an integral t takes just 0 and 1
    template<Floating T, std::size_t N>
    [[nodiscard]] constexpr auto lerp(const Vec<T, N> &a, const Vec<T, N> &b, T t) noexcept -> Vec<T, N> {
        return a + (b - a) * t;
    }

    template<Number T, std::size_t N>
    [[nodiscard]] constexpr auto min(const Vec<T, N> &a, const Vec<T, N> &b) noexcept -> Vec<T, N> {
        Vec<T, N> res;
        for (std::size_t i = 0; i < N; ++i) {
            res[i] = a[i] < b[i] ? a[i] : b[i];
        }
        return res;
    }

    template<Number T, std::size_t N>
    [[nodiscard]] constexpr auto max(const Vec<T, N> &a, const Vec<T, N> &b) noexcept -> Vec<T, N> {
        Vec<T, N> res;
        for (std::size_t i = 0; i < N; ++i) {
            res[i] = a[i] < b[i] ? b[i] : a[i];
        }
        return res;
    }

    template<Number T, std::size_t N>
    [[nodiscard]] constexpr auto abs(const Vec<T, N> &v) noexcept -> Vec<T, N> {
        Vec<T, N> res;
        for (std::size_t i = 0; i < N; ++i) {
            res[i] = v[i] < T{0} ? -v[i] : v[i];
        }
        return res;
    }

    template<Number T, std::size_t N>
    [[nodiscard]] constexpr auto clamp(
        const Vec<T, N> &v,
        const Vec<T, N> &lo, const Vec<T, N> &hi
    ) noexcept -> Vec<T, N> {
        return min(max(v, lo), hi);
    }

    template<Number T, std::size_t N>
    [[nodiscard]] constexpr auto clamp(const Vec<T, N> &v, T lo, T hi) noexcept -> Vec<T, N> {
        return clamp(v, Vec<T, N>{lo}, Vec<T, N>{hi});
    }

    template<Floating T, std::size_t N>
    [[nodiscard]] constexpr auto saturate(const Vec<T, N> &v) noexcept -> Vec<T, N> {
        return clamp(v, T{0}, T{1});
    }

    // mixed tolerance: absolute near zero, relative at large magnitudes
    template<Floating T>
    [[nodiscard]] constexpr auto approx_eq(
        T a, T b,
        T eps = std::numeric_limits<T>::epsilon() * T{8}
    ) noexcept -> bool {
        const T d = a < b ? b - a : a - b;
        const T fa = a < T{0} ? -a : a;
        const T fb = b < T{0} ? -b : b;
        const T m = fa < fb ? fb : fa;
        return d <= eps * (m < T{1} ? T{1} : m);
    }

    template<Floating T, std::size_t N>
    [[nodiscard]] constexpr auto approx_eq(
        const Vec<T, N> &a, const Vec<T, N> &b,
        T eps = std::numeric_limits<T>::epsilon() * T{8}
    ) noexcept -> bool {
        for (std::size_t i = 0; i < N; ++i) {
            if (!approx_eq(a[i], b[i], eps)) {
                return false;
            }
        }
        return true;
    }

    template<Floating T>
    [[nodiscard]] constexpr auto radians(T deg) noexcept -> T {
        return deg * (std::numbers::pi_v<T> / T{180});
    }

    template<Floating T>
    [[nodiscard]] constexpr auto degrees(T rad) noexcept -> T {
        return rad * (T{180} / std::numbers::pi_v<T>);
    }

    using Vec2i = Vec<std::int32_t, 2>;
    using Vec3i = Vec<std::int32_t, 3>;
    using Vec4i = Vec<std::int32_t, 4>;
    using Vec2f = Vec<float, 2>;
    using Vec3f = Vec<float, 3>;
    using Vec4f = Vec<float, 4>;
    using Vec2d = Vec<double, 2>;
    using Vec3d = Vec<double, 3>;
    using Vec4d = Vec<double, 4>;

    static_assert(sizeof(Vec2i) == sizeof(std::int32_t) * 2);
    static_assert(sizeof(Vec3i) == sizeof(std::int32_t) * 3);
    static_assert(sizeof(Vec4i) == sizeof(std::int32_t) * 4);
    static_assert(sizeof(Vec2f) == sizeof(float) * 2);
    static_assert(sizeof(Vec3f) == sizeof(float) * 3);
    static_assert(sizeof(Vec4f) == sizeof(float) * 4);
    static_assert(sizeof(Vec2d) == sizeof(double) * 2);
    static_assert(sizeof(Vec3d) == sizeof(double) * 3);
    static_assert(sizeof(Vec4d) == sizeof(double) * 4);
}

// ============================================================================
// mat
// ============================================================================

// conventions of this section, none of them changeable in isolation:
//   - column-major storage, column vectors, M * v
//   - right-handed view space, look_at faces -Z
//   - depth in [0, 1] (native to Vulkan and D3D; GL needs glClipControl, 4.5+)
//   - angles in radians

namespace tla {
    template<Number T, std::size_t R, std::size_t C>
        requires (R > 0 && C > 0)
    class Mat {
        // column-major: m_cols[c] is column c
        std::array<Vec<T, R>, C> m_cols{};

    public:
        // zero, NOT identity: glm's mat4(1.0f) is a known trap
        constexpr Mat() noexcept = default;

        template<typename... Cols>
            requires (sizeof...(Cols) == C && (std::convertible_to<Cols, Vec<T, R> > && ...))
        constexpr Mat(Cols... cols) noexcept : m_cols{static_cast<Vec<T, R>>(cols)...} {
        }

        [[nodiscard]] static constexpr auto identity() noexcept -> Mat requires (R == C) {
            Mat res;
            for (std::size_t i = 0; i < R; ++i) {
                res.m_cols[i][i] = T{1};
            }
            return res;
        }

        [[nodiscard]] static constexpr auto diagonal(const Vec<T, R> &d) noexcept -> Mat requires (R == C) {
            Mat res;
            for (std::size_t i = 0; i < R; ++i) {
                res.m_cols[i][i] = d[i];
            }
            return res;
        }

        [[nodiscard]] constexpr auto col(this auto &&self, std::size_t c) noexcept -> auto & {
            return self.m_cols[c];
        }

        // by value: a row is not contiguous under column-major storage
        [[nodiscard]] constexpr auto row(std::size_t r) const noexcept -> Vec<T, C> {
            Vec<T, C> res;
            for (std::size_t c = 0; c < C; ++c) {
                res[c] = m_cols[c][r];
            }
            return res;
        }

        // math index order. no single-argument [] on purpose: that one
        // is what breeds the M[c][r] confusion
        [[nodiscard]] constexpr auto operator[](this auto &&self, std::size_t r, std::size_t c) noexcept -> auto & {
            return self.m_cols[c][r];
        }

        // contiguity is a hard invariant: GPU upload relies on it
        [[nodiscard]] constexpr auto data(this auto &&self) noexcept -> auto * {
            static_assert(sizeof(Mat) == sizeof(T) * R * C, "padded columns: storage is not contiguous");
            return self.m_cols[0].data();
        }

        // Vec::xyz() analogue: truncate to the top-left corner
        template<std::size_t R2, std::size_t C2>
            requires (R2 > 0 && C2 > 0 && R2 <= R && C2 <= C)
        [[nodiscard]] constexpr auto top_left() const noexcept -> Mat<T, R2, C2> {
            Mat<T, R2, C2> res;
            for (std::size_t c = 0; c < C2; ++c) {
                for (std::size_t r = 0; r < R2; ++r) {
                    res[r, c] = m_cols[c][r];
                }
            }
            return res;
        }

        constexpr auto operator+=(const Mat &m) noexcept -> Mat & {
            for (std::size_t c = 0; c < C; ++c) {
                m_cols[c] += m.m_cols[c];
            }
            return *this;
        }

        constexpr auto operator-=(const Mat &m) noexcept -> Mat & {
            for (std::size_t c = 0; c < C; ++c) {
                m_cols[c] -= m.m_cols[c];
            }
            return *this;
        }

        constexpr auto operator*=(T s) noexcept -> Mat & {
            for (std::size_t c = 0; c < C; ++c) {
                m_cols[c] *= s;
            }
            return *this;
        }

        constexpr auto operator/=(T s) noexcept -> Mat & {
            for (std::size_t c = 0; c < C; ++c) {
                m_cols[c] /= s;
            }
            return *this;
        }

        constexpr auto operator*=(const Mat &m) noexcept -> Mat & requires (R == C) {
            *this = *this * m;
            return *this;
        }

        [[nodiscard]] friend constexpr auto operator+(Mat lhs, const Mat &rhs) noexcept -> Mat {
            lhs += rhs;
            return lhs;
        }

        [[nodiscard]] friend constexpr auto operator-(Mat lhs, const Mat &rhs) noexcept -> Mat {
            lhs -= rhs;
            return lhs;
        }

        [[nodiscard]] friend constexpr auto operator*(Mat lhs, T rhs) noexcept -> Mat {
            lhs *= rhs;
            return lhs;
        }

        [[nodiscard]] friend constexpr auto operator*(T lhs, Mat rhs) noexcept -> Mat {
            rhs *= lhs;
            return rhs;
        }

        [[nodiscard]] friend constexpr auto operator/(Mat lhs, T rhs) noexcept -> Mat {
            lhs /= rhs;
            return lhs;
        }

        [[nodiscard]] constexpr auto operator-() const noexcept -> Mat {
            Mat res;
            for (std::size_t c = 0; c < C; ++c) {
                res.m_cols[c] = -m_cols[c];
            }
            return res;
        }

        [[nodiscard]] constexpr auto operator==(const Mat &) const noexcept -> bool = default;
    };
}

namespace tla::detail {
    // inverse == adjugate / det. shared so that inverse and inverse_or
    // do not each roll their own copy
    template<Number T, std::size_t N>
        requires (N <= 4)
    [[nodiscard]] constexpr auto adjugate(const Mat<T, N, N> &m) noexcept -> Mat<T, N, N> {
        Mat<T, N, N> r;
        if constexpr (N == 1) {
            r[0, 0] = T{1};
        } else if constexpr (N == 2) {
            r[0, 0] = m[1, 1];
            r[0, 1] = -m[0, 1];
            r[1, 0] = -m[1, 0];
            r[1, 1] = m[0, 0];
        } else if constexpr (N == 3) {
            r[0, 0] = m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1];
            r[0, 1] = m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2];
            r[0, 2] = m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1];
            r[1, 0] = m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2];
            r[1, 1] = m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0];
            r[1, 2] = m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2];
            r[2, 0] = m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0];
            r[2, 1] = m[0, 1] * m[2, 0] - m[0, 0] * m[2, 1];
            r[2, 2] = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0];
        } else {
            const T s0 = m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1];
            const T s1 = m[0, 0] * m[1, 2] - m[1, 0] * m[0, 2];
            const T s2 = m[0, 0] * m[1, 3] - m[1, 0] * m[0, 3];
            const T s3 = m[0, 1] * m[1, 2] - m[1, 1] * m[0, 2];
            const T s4 = m[0, 1] * m[1, 3] - m[1, 1] * m[0, 3];
            const T s5 = m[0, 2] * m[1, 3] - m[1, 2] * m[0, 3];
            const T c5 = m[2, 2] * m[3, 3] - m[3, 2] * m[2, 3];
            const T c4 = m[2, 1] * m[3, 3] - m[3, 1] * m[2, 3];
            const T c3 = m[2, 1] * m[3, 2] - m[3, 1] * m[2, 2];
            const T c2 = m[2, 0] * m[3, 3] - m[3, 0] * m[2, 3];
            const T c1 = m[2, 0] * m[3, 2] - m[3, 0] * m[2, 2];
            const T c0 = m[2, 0] * m[3, 1] - m[3, 0] * m[2, 1];

            r[0, 0] = m[1, 1] * c5 - m[1, 2] * c4 + m[1, 3] * c3;
            r[0, 1] = -m[0, 1] * c5 + m[0, 2] * c4 - m[0, 3] * c3;
            r[0, 2] = m[3, 1] * s5 - m[3, 2] * s4 + m[3, 3] * s3;
            r[0, 3] = -m[2, 1] * s5 + m[2, 2] * s4 - m[2, 3] * s3;
            r[1, 0] = -m[1, 0] * c5 + m[1, 2] * c2 - m[1, 3] * c1;
            r[1, 1] = m[0, 0] * c5 - m[0, 2] * c2 + m[0, 3] * c1;
            r[1, 2] = -m[3, 0] * s5 + m[3, 2] * s2 - m[3, 3] * s1;
            r[1, 3] = m[2, 0] * s5 - m[2, 2] * s2 + m[2, 3] * s1;
            r[2, 0] = m[1, 0] * c4 - m[1, 1] * c2 + m[1, 3] * c0;
            r[2, 1] = -m[0, 0] * c4 + m[0, 1] * c2 - m[0, 3] * c0;
            r[2, 2] = m[3, 0] * s4 - m[3, 1] * s2 + m[3, 3] * s0;
            r[2, 3] = -m[2, 0] * s4 + m[2, 1] * s2 - m[2, 3] * s0;
            r[3, 0] = -m[1, 0] * c3 + m[1, 1] * c1 - m[1, 2] * c0;
            r[3, 1] = m[0, 0] * c3 - m[0, 1] * c1 + m[0, 2] * c0;
            r[3, 2] = -m[3, 0] * s3 + m[3, 1] * s1 - m[3, 2] * s0;
            r[3, 3] = m[2, 0] * s3 - m[2, 1] * s1 + m[2, 2] * s0;
        }
        return r;
    }
}

namespace tla {
    // a linear combination of columns is the definition of M * v
    template<Number T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr auto operator*(const Mat<T, R, C> &m, const Vec<T, C> &v) noexcept -> Vec<T, R> {
        Vec<T, R> res;
        for (std::size_t c = 0; c < C; ++c) {
            res += m.col(c) * v[c];
        }
        return res;
    }

    // each column of the result is one matvec.
    // NOTE: * on matrices is the matrix product, not componentwise as it is on Vec.
    template<Number T, std::size_t R, std::size_t K, std::size_t C>
    [[nodiscard]] constexpr auto operator*(const Mat<T, R, K> &a, const Mat<T, K, C> &b) noexcept -> Mat<T, R, C> {
        Mat<T, R, C> res;
        for (std::size_t c = 0; c < C; ++c) {
            res.col(c) = a * b.col(c);
        }
        return res;
    }

    template<Number T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr auto transpose(const Mat<T, R, C> &m) noexcept -> Mat<T, C, R> {
        Mat<T, C, R> res;
        for (std::size_t c = 0; c < C; ++c) {
            for (std::size_t r = 0; r < R; ++r) {
                res[c, r] = m[r, c];
            }
        }
        return res;
    }

    // graphics never goes past 4, so explicit formulas rather than Gauss
    template<Number T, std::size_t N>
        requires (N <= 4)
    [[nodiscard]] constexpr auto determinant(const Mat<T, N, N> &m) noexcept -> T {
        if constexpr (N == 1) {
            return m[0, 0];
        } else if constexpr (N == 2) {
            return m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0];
        } else if constexpr (N == 3) {
            return m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
                   - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
                   + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]);
        } else {
            const T s0 = m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1];
            const T s1 = m[0, 0] * m[1, 2] - m[1, 0] * m[0, 2];
            const T s2 = m[0, 0] * m[1, 3] - m[1, 0] * m[0, 3];
            const T s3 = m[0, 1] * m[1, 2] - m[1, 1] * m[0, 2];
            const T s4 = m[0, 1] * m[1, 3] - m[1, 1] * m[0, 3];
            const T s5 = m[0, 2] * m[1, 3] - m[1, 2] * m[0, 3];
            const T c5 = m[2, 2] * m[3, 3] - m[3, 2] * m[2, 3];
            const T c4 = m[2, 1] * m[3, 3] - m[3, 1] * m[2, 3];
            const T c3 = m[2, 1] * m[3, 2] - m[3, 1] * m[2, 2];
            const T c2 = m[2, 0] * m[3, 3] - m[3, 0] * m[2, 3];
            const T c1 = m[2, 0] * m[3, 2] - m[3, 0] * m[2, 2];
            const T c0 = m[2, 0] * m[3, 1] - m[3, 0] * m[2, 1];
            return s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
        }
    }

    // same pair as normalize/normalize_or: the degenerate case is caught by an assert
    template<Floating T, std::size_t N>
        requires (N <= 4)
    [[nodiscard]] constexpr auto inverse(const Mat<T, N, N> &m) noexcept -> Mat<T, N, N> {
        const T det = determinant(m);
        assert(det != T{0} && "inverse of a singular matrix");
        return detail::adjugate(m) * (T{1} / det);
    }

    template<Floating T, std::size_t N>
        requires (N <= 4)
    [[nodiscard]] constexpr auto inverse_or(
        const Mat<T, N, N> &m,
        const Mat<T, N, N> &fallback
    ) noexcept -> Mat<T, N, N> {
        const T det = determinant(m);
        return det != T{0} ? detail::adjugate(m) * (T{1} / det) : fallback;
    }

    // lighting under non-uniform scale needs this instead of the model matrix
    template<Floating T>
    [[nodiscard]] constexpr auto normal_matrix(const Mat<T, 4, 4> &m) noexcept -> Mat<T, 3, 3> {
        return transpose(inverse(m.template top_left<3, 3>()));
    }

    template<Floating T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr auto approx_eq(
        const Mat<T, R, C> &a, const Mat<T, R, C> &b,
        T eps = std::numeric_limits<T>::epsilon() * T{8}
    ) noexcept -> bool {
        for (std::size_t c = 0; c < C; ++c) {
            if (!approx_eq(a.col(c), b.col(c), eps)) {
                return false;
            }
        }
        return true;
    }

    template<Number T>
    [[nodiscard]] constexpr auto translation(const Vec<T, 3> &t) noexcept -> Mat<T, 4, 4> {
        auto res = Mat<T, 4, 4>::identity();
        res.col(3) = t.xyzw(T{1});
        return res;
    }

    // uniform scale is written scale(Vec3f{s}). no scalar overload on purpose:
    // scale(2.f) would read ambiguously
    template<Number T>
    [[nodiscard]] constexpr auto scale(const Vec<T, 3> &s) noexcept -> Mat<T, 4, 4> {
        return Mat<T, 4, 4>::diagonal(s.xyzw(T{1}));
    }

    template<Floating T>
    [[nodiscard]] auto rotation_x(T a) noexcept -> Mat<T, 4, 4> {
        const T c = std::cos(a);
        const T s = std::sin(a);
        return {
            Vec<T, 4>{T{1}, T{0}, T{0}, T{0}},
            Vec<T, 4>{T{0}, c, s, T{0}},
            Vec<T, 4>{T{0}, -s, c, T{0}},
            Vec<T, 4>{T{0}, T{0}, T{0}, T{1}},
        };
    }

    template<Floating T>
    [[nodiscard]] auto rotation_y(T a) noexcept -> Mat<T, 4, 4> {
        const T c = std::cos(a);
        const T s = std::sin(a);
        return {
            Vec<T, 4>{c, T{0}, -s, T{0}},
            Vec<T, 4>{T{0}, T{1}, T{0}, T{0}},
            Vec<T, 4>{s, T{0}, c, T{0}},
            Vec<T, 4>{T{0}, T{0}, T{0}, T{1}},
        };
    }

    template<Floating T>
    [[nodiscard]] auto rotation_z(T a) noexcept -> Mat<T, 4, 4> {
        const T c = std::cos(a);
        const T s = std::sin(a);
        return {
            Vec<T, 4>{c, s, T{0}, T{0}},
            Vec<T, 4>{-s, c, T{0}, T{0}},
            Vec<T, 4>{T{0}, T{0}, T{1}, T{0}},
            Vec<T, 4>{T{0}, T{0}, T{0}, T{1}},
        };
    }

    // Rodrigues' formula.
    template<Floating T>
    [[nodiscard]] auto rotation(const Vec<T, 3> &axis, T a) noexcept -> Mat<T, 4, 4> {
        // the tolerance is generous on purpose: an axis that is wrong is wrong by
        // a lot, while a legitimately normalized one carries a few ulps of drift
        assert(approx_eq(length_sq(axis), T{1}, std::numeric_limits<T>::epsilon() * T{64})
            && "rotation about a non-normalized axis");

        const T c = std::cos(a);
        const T s = std::sin(a);
        const T k = T{1} - c;
        const T x = axis.x();
        const T y = axis.y();
        const T z = axis.z();
        return {
            Vec<T, 4>{c + x * x * k, y * x * k + z * s, z * x * k - y * s, T{0}},
            Vec<T, 4>{x * y * k - z * s, c + y * y * k, z * y * k + x * s, T{0}},
            Vec<T, 4>{x * z * k + y * s, y * z * k - x * s, c + z * z * k, T{0}},
            Vec<T, 4>{T{0}, T{0}, T{0}, T{1}},
        };
    }

    template<Floating T>
    [[nodiscard]] auto look_at(
        const Vec<T, 3> &eye,
        const Vec<T, 3> &center,
        const Vec<T, 3> &up
    ) noexcept -> Mat<T, 4, 4> {
        const auto f = normalize(center - eye);
        const auto s = normalize(cross(f, up));
        const auto u = cross(s, f);
        return {
            Vec<T, 4>{s.x(), u.x(), -f.x(), T{0}},
            Vec<T, 4>{s.y(), u.y(), -f.y(), T{0}},
            Vec<T, 4>{s.z(), u.z(), -f.z(), T{0}},
            Vec<T, 4>{-dot(s, eye), -dot(u, eye), dot(f, eye), T{1}},
        };
    }

    // fovy is the full vertical field of view in radians.
    template<Floating T>
    [[nodiscard]] auto perspective(T fovy, T aspect, T z_near, T z_far) noexcept -> Mat<T, 4, 4> {
        assert(fovy > T{0} && fovy < std::numbers::pi_v<T> && "perspective: fovy is outside (0, pi)");
        assert(aspect > T{0} && "perspective: aspect is not positive");
        assert(z_near > T{0} && "perspective: z_near is not positive");
        assert(z_far > z_near && "perspective: z_far is not beyond z_near");

        const T th = std::tan(fovy / T{2});
        Mat<T, 4, 4> res;
        res[0, 0] = T{1} / (aspect * th);
        res[1, 1] = T{1} / th;
        res[2, 2] = z_far / (z_near - z_far);
        res[2, 3] = -(z_far * z_near) / (z_far - z_near);
        res[3, 2] = T{-1};
        return res;
    }

    template<Floating T>
    [[nodiscard]] constexpr auto ortho(T l, T r, T b, T t, T z_near, T z_far) noexcept -> Mat<T, 4, 4> {
        // only non-degeneracy is required. an inverted range is the usual way to
        // flip an axis -- ortho(0, w, h, 0, ...) is y-down screen space
        assert(r != l && t != b && z_far != z_near && "ortho: degenerate range");

        Mat<T, 4, 4> res;
        res[0, 0] = T{2} / (r - l);
        res[1, 1] = T{2} / (t - b);
        res[2, 2] = T{-1} / (z_far - z_near);
        res[0, 3] = -(r + l) / (r - l);
        res[1, 3] = -(t + b) / (t - b);
        res[2, 3] = -z_near / (z_far - z_near);
        res[3, 3] = T{1};
        return res;
    }

    // NDC -> screen. depth is already in [0, 1] and is left alone.
    template<Floating T>
    [[nodiscard]] constexpr auto viewport(T x, T y, T w, T h) noexcept -> Mat<T, 4, 4> {
        // a negative extent flips an axis on purpose; zero is always a bug
        assert(w != T{0} && h != T{0} && "viewport: zero extent");

        Mat<T, 4, 4> res;
        res[0, 0] = w / T{2};
        res[1, 1] = h / T{2};
        res[2, 2] = T{1};
        res[0, 3] = x + w / T{2};
        res[1, 3] = y + h / T{2};
        res[3, 3] = T{1};
        return res;
    }

    using Mat2f = Mat<float, 2, 2>;
    using Mat3f = Mat<float, 3, 3>;
    using Mat4f = Mat<float, 4, 4>;
    using Mat2d = Mat<double, 2, 2>;
    using Mat3d = Mat<double, 3, 3>;
    using Mat4d = Mat<double, 4, 4>;

    static_assert(sizeof(Mat2f) == sizeof(float) * 4);
    static_assert(sizeof(Mat3f) == sizeof(float) * 9);
    static_assert(sizeof(Mat4f) == sizeof(float) * 16);
    static_assert(sizeof(Mat2d) == sizeof(double) * 4);
    static_assert(sizeof(Mat3d) == sizeof(double) * 9);
    static_assert(sizeof(Mat4d) == sizeof(double) * 16);
}

// ============================================================================
// structure bindings
// ============================================================================

template<tla::Number T, std::size_t N>
    requires (N > 0)
struct std::tuple_size<tla::Vec<T, N> > : std::integral_constant<std::size_t, N> {
};

template<std::size_t I, tla::Number T, std::size_t N>
    requires (N > 0)
struct std::tuple_element<I, tla::Vec<T, N> > {
    using type = T;
};

// ============================================================================
// formatting
// ============================================================================

template<tla::Number T, std::size_t N>
    requires (N > 0)
struct std::formatter<tla::Vec<T, N>, char> : std::formatter<std::string, char> {
    auto format(const tla::Vec<T, N> &v, auto &ctx) const {
        std::string s = "(";
        for (std::size_t i = 0; i < N; ++i) {
            if (i != 0) s += ", ";
            s += std::format("{}", v[i]);
        }
        s += ')';
        return std::formatter<std::string, char>::format(s, ctx);
    }
};

// by rows and width-aligned: a matrix is read the way it is written in a formula,
// and column-major is a storage detail there is no reason to see while debugging
template<tla::Number T, std::size_t R, std::size_t C>
    requires (R > 0 && C > 0)
struct std::formatter<tla::Mat<T, R, C>, char> : std::formatter<std::string, char> {
    auto format(const tla::Mat<T, R, C> &m, auto &ctx) const {
        std::array<std::string, R * C> cell;
        std::size_t w = 0;
        for (std::size_t r = 0; r < R; ++r) {
            for (std::size_t c = 0; c < C; ++c) {
                auto &e = cell[r * C + c];
                e = std::format("{}", m[r, c]);
                w = w < e.size() ? e.size() : w;
            }
        }

        std::string s;
        for (std::size_t r = 0; r < R; ++r) {
            if (r != 0) {
                s += '\n';
            }
            s += '[';
            for (std::size_t c = 0; c < C; ++c) {
                const auto &e = cell[r * C + c];
                s += ' ';
                s.append(w - e.size(), ' ');
                s += e;
            }
            s += " ]";
        }
        return std::formatter<std::string, char>::format(s, ctx);
    }
};

namespace tla {
    template<Number T, std::size_t N>
        requires (N > 0)
    auto operator<<(std::ostream &os, const Vec<T, N> &v) -> std::ostream & {
        return os << std::format("{}", v);
    }

    template<Number T, std::size_t R, std::size_t C>
        requires (R > 0 && C > 0)
    auto operator<<(std::ostream &os, const Mat<T, R, C> &m) -> std::ostream & {
        return os << std::format("{}", m);
    }
}
