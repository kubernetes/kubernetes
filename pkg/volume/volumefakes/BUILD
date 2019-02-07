load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["fake_kubelet_watchable_volume_plugin.go"],
    importpath = "k8s.io/kubernetes/pkg/volume/volumefakes",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubelet/pluginmanager/cache:go_default_library",
        "//pkg/volume:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
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
