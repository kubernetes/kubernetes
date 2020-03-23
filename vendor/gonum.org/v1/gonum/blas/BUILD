load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "blas.go",
        "doc.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gonum.org/v1/gonum/blas",
    importpath = "gonum.org/v1/gonum/blas",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//vendor/gonum.org/v1/gonum/blas/blas64:all-srcs",
        "//vendor/gonum.org/v1/gonum/blas/cblas128:all-srcs",
        "//vendor/gonum.org/v1/gonum/blas/gonum:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
