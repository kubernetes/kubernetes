package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)
load("//staging/src/k8s.io/component-base/version:def.bzl", "version_x_defs")

go_binary(
    name = "kube-controller-manager",
    embed = [":go_default_library"],
    pure = "off",
    x_defs = version_x_defs(),
)

go_library(
    name = "go_default_library",
    srcs = ["controller-manager.go"],
    importpath = "k8s.io/kubernetes/cmd/kube-controller-manager",
    deps = [
        "//cmd/kube-controller-manager/app:go_default_library",
        "//staging/src/k8s.io/component-base/logs:go_default_library",
        "//staging/src/k8s.io/component-base/metrics/prometheus/clientgo:go_default_library",
        "//staging/src/k8s.io/component-base/metrics/prometheus/version:go_default_library",
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
        "//cmd/kube-controller-manager/app:all-srcs",
    ],
    tags = ["automanaged"],
)
