load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["workspace.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/bazelbuild/buildtools/wspace",
    importpath = "github.com/bazelbuild/buildtools/wspace",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/bazelbuild/buildtools/build:go_default_library"],
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
