load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["fake_plugin_watcher.go"],
    importpath = "k8s.io/kubernetes/pkg/kubelet/kubeletfakes",
    visibility = ["//visibility:public"],
    deps = ["//pkg/kubelet/util/pluginwatcher:go_default_library"],
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
