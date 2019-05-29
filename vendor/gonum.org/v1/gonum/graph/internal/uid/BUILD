load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["uid.go"],
    importmap = "k8s.io/kubernetes/vendor/gonum.org/v1/gonum/graph/internal/uid",
    importpath = "gonum.org/v1/gonum/graph/internal/uid",
    visibility = ["//vendor/gonum.org/v1/gonum/graph:__subpackages__"],
    deps = ["//vendor/gonum.org/v1/gonum/graph/internal/set:go_default_library"],
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
