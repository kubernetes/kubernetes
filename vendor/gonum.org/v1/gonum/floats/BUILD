load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "floats.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gonum.org/v1/gonum/floats",
    importpath = "gonum.org/v1/gonum/floats",
    visibility = ["//visibility:public"],
    deps = ["//vendor/gonum.org/v1/gonum/internal/asm/f64:go_default_library"],
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
