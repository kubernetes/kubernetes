load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["main.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/cloud-provider/sample",
    importpath = "k8s.io/cloud-provider/sample",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/cloud-provider:go_default_library",
        "//staging/src/k8s.io/cloud-provider/app:go_default_library",
        "//staging/src/k8s.io/cloud-provider/options:go_default_library",
        "//staging/src/k8s.io/component-base/cli/flag:go_default_library",
        "//staging/src/k8s.io/component-base/logs:go_default_library",
        "//staging/src/k8s.io/component-base/metrics/prometheus/clientgo:go_default_library",
        "//staging/src/k8s.io/component-base/metrics/prometheus/version:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
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
    visibility = ["//visibility:public"],
)
