load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "lapack.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gonum.org/v1/gonum/lapack",
    importpath = "gonum.org/v1/gonum/lapack",
    visibility = ["//visibility:public"],
    deps = ["//vendor/gonum.org/v1/gonum/blas:go_default_library"],
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
        "//vendor/gonum.org/v1/gonum/lapack/gonum:all-srcs",
        "//vendor/gonum.org/v1/gonum/lapack/lapack64:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
