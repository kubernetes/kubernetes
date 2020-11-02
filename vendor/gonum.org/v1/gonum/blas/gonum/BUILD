load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "dgemm.go",
        "doc.go",
        "errors.go",
        "gemv.go",
        "gonum.go",
        "level1cmplx128.go",
        "level1cmplx64.go",
        "level1float32.go",
        "level1float32_dsdot.go",
        "level1float32_sdot.go",
        "level1float32_sdsdot.go",
        "level1float64.go",
        "level1float64_ddot.go",
        "level2cmplx128.go",
        "level2cmplx64.go",
        "level2float32.go",
        "level2float64.go",
        "level3cmplx128.go",
        "level3cmplx64.go",
        "level3float32.go",
        "level3float64.go",
        "sgemm.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gonum.org/v1/gonum/blas/gonum",
    importpath = "gonum.org/v1/gonum/blas/gonum",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/gonum.org/v1/gonum/blas:go_default_library",
        "//vendor/gonum.org/v1/gonum/internal/asm/c128:go_default_library",
        "//vendor/gonum.org/v1/gonum/internal/asm/c64:go_default_library",
        "//vendor/gonum.org/v1/gonum/internal/asm/f32:go_default_library",
        "//vendor/gonum.org/v1/gonum/internal/asm/f64:go_default_library",
        "//vendor/gonum.org/v1/gonum/internal/cmplx64:go_default_library",
        "//vendor/gonum.org/v1/gonum/internal/math32:go_default_library",
    ],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
