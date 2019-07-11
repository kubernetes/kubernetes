load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["verifydependencies.go"],
    importpath = "k8s.io/kubernetes/cmd/verifydependencies",
    visibility = ["//visibility:private"],
    deps = ["//vendor/gopkg.in/yaml.v2:go_default_library"],
)

go_binary(
    name = "verifydependencies",
    embed = [":go_default_library"],
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
