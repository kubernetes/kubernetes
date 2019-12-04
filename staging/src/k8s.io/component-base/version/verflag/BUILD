package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)
load("//staging/src/k8s.io/component-base/version:def.bzl", "version_x_defs")

go_library(
    name = "go_default_library",
    srcs = ["verflag.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/component-base/version/verflag",
    importpath = "k8s.io/component-base/version/verflag",
    deps = [
        "//staging/src/k8s.io/component-base/version:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
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
)
