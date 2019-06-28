package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)
load("//staging/src/k8s.io/component-base/version:def.bzl", "version_x_defs")

go_binary(
    name = "kube-proxy",
    embed = [":go_default_library"],
    pure = "off",
    x_defs = version_x_defs(),
)

go_library(
    name = "go_default_library",
    srcs = ["proxy.go"],
    importpath = "k8s.io/kubernetes/cmd/kube-proxy",
    deps = [
        "//cmd/kube-proxy/app:go_default_library",
        "//staging/src/k8s.io/component-base/cli/flag:go_default_library",
        "//staging/src/k8s.io/component-base/logs:go_default_library",
        "//staging/src/k8s.io/component-base/metrics/prometheus/restclient:go_default_library",
        "//staging/src/k8s.io/component-base/metrics/prometheus/version:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//cmd/kube-proxy/app:all-srcs",
    ],
    tags = ["automanaged"],
)
