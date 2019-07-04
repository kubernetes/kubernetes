load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "math.go",
        "signbit.go",
        "sqrt.go",
        "sqrt_amd64.go",
        "sqrt_amd64.s",
    ],
    importmap = "k8s.io/kubernetes/vendor/gonum.org/v1/gonum/internal/math32",
    importpath = "gonum.org/v1/gonum/internal/math32",
    visibility = ["//vendor/gonum.org/v1/gonum:__subpackages__"],
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
