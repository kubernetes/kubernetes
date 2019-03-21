load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "cmplx.go",
        "dgemm.go",
        "doc.go",
        "gemv.go",
        "gonum.go",
        "level1cmplx128.go",
        "level1double.go",
        "level1double_ddot.go",
        "level1single.go",
        "level1single_dsdot.go",
        "level1single_sdot.go",
        "level1single_sdsdot.go",
        "level2cmplx128.go",
        "level2double.go",
        "level2single.go",
        "level3double.go",
        "level3single.go",
        "sgemm.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gonum.org/v1/gonum/blas/gonum",
    importpath = "gonum.org/v1/gonum/blas/gonum",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/gonum.org/v1/gonum/blas:go_default_library",
        "//vendor/gonum.org/v1/gonum/internal/asm/c128:go_default_library",
        "//vendor/gonum.org/v1/gonum/internal/asm/f32:go_default_library",
        "//vendor/gonum.org/v1/gonum/internal/asm/f64:go_default_library",
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
