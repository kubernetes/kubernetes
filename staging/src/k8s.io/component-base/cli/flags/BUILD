load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["flags.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/component-base/cli/flags",
    importpath = "k8s.io/component-base/cli/flags",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/spf13/pflag:go_default_library"],
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
        "//staging/src/k8s.io/component-base/cli/flags/example:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
