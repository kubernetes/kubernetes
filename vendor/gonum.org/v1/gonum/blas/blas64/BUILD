load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "blas64.go",
        "conv.go",
        "conv_symmetric.go",
        "doc.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gonum.org/v1/gonum/blas/blas64",
    importpath = "gonum.org/v1/gonum/blas/blas64",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/gonum.org/v1/gonum/blas:go_default_library",
        "//vendor/gonum.org/v1/gonum/blas/gonum:go_default_library",
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
