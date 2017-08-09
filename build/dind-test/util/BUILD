package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)
load("//pkg/version:def.bzl", "version_x_defs")

go_binary(
    name = "k8sversion",
    embed = [":go_default_library"],
    importpath = "k8s.io/kubernetes/build/nested-cluster/util",
    pure = "on",
    x_defs = version_x_defs(),
)

go_library(
    name = "go_default_library",
    srcs = ["version.go"],
    importpath = "k8s.io/kubernetes/build/nested-cluster/util",
    deps = ["//pkg/version:go_default_library"],
)

go_binary(
    name = "util",
    embed = [":go_default_library"],
    importpath = "k8s.io/kubernetes/build/nested-cluster/util",
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
