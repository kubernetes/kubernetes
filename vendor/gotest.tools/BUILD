load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["pkg.go"],
    importmap = "k8s.io/kubernetes/vendor/gotest.tools",
    importpath = "gotest.tools",
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
        "//vendor/gotest.tools/gotestsum:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
