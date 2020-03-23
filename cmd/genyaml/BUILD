package(default_visibility = ["//visibility:public"])

load(
    "//build:go.bzl",
    go_binary = "go_binary_conditional_pure",
)
load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_binary(
    name = "genyaml",
    embed = [":go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = ["gen_kubectl_yaml.go"],
    importpath = "k8s.io/kubernetes/cmd/genyaml",
    deps = [
        "//cmd/genutils:go_default_library",
        "//pkg/kubectl/cmd:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/gopkg.in/yaml.v2:go_default_library",
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
