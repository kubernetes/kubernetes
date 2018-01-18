package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "explorer",
    embed = [":go_default_library"],
    importpath = "k8s.io/kubernetes/examples/explorer",
)

go_library(
    name = "go_default_library",
    srcs = ["explorer.go"],
    importpath = "k8s.io/kubernetes/examples/explorer",
    deps = ["//vendor/github.com/davecgh/go-spew/spew:go_default_library"],
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
