package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "clicheck",
    embed = [":go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = ["check_cli_conventions.go"],
    importpath = "k8s.io/kubernetes/cmd/clicheck",
    deps = [
        "//pkg/kubectl/cmd:go_default_library",
        "//staging/src/k8s.io/kubectl/pkg/cmd/util/sanity:go_default_library",
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
