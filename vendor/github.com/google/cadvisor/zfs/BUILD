load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["watcher.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/google/cadvisor/zfs",
    importpath = "github.com/google/cadvisor/zfs",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/mistifyio/go-zfs:go_default_library",
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
