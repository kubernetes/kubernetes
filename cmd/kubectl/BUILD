load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)
load("//pkg/version:def.bzl", "version_x_defs")

go_binary(
    name = "kubectl",
    embed = [":go_default_library"],
    importpath = "k8s.io/kubernetes/cmd/kubectl",
    pure = "on",
    visibility = ["//visibility:public"],
    x_defs = version_x_defs(),
)

go_library(
    name = "go_default_library",
    srcs = ["kubectl.go"],
    importpath = "k8s.io/kubernetes/cmd/kubectl",
    visibility = ["//visibility:private"],
    deps = ["//cmd/kubectl/app:go_default_library"],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//cmd/kubectl/app:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = [
        "//build/visible_to:cmd_kubectl_CONSUMERS",
    ],
)
