package(default_visibility = ["//visibility:public"])

load(
    "//build:go.bzl",
    go_binary = "go_binary_conditional_pure",
)
load("@io_bazel_rules_go//go:def.bzl", "go_library")
load("//pkg/version:def.bzl", "version_x_defs")

go_binary(
    name = "kubelet",
    embed = [":go_default_library"],
    x_defs = version_x_defs(),
)

go_library(
    name = "go_default_library",
    srcs = ["kubelet.go"],
    importpath = "k8s.io/kubernetes/cmd/kubelet",
    deps = [
        "//cmd/kubelet/app:go_default_library",
        "//pkg/client/metrics/prometheus:go_default_library",
        "//pkg/version/prometheus:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server:go_default_library",
        "//staging/src/k8s.io/component-base/logs:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//cmd/kubelet/app:all-srcs",
    ],
    tags = ["automanaged"],
)
