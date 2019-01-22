load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["rkt.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/google/cadvisor/manager/watcher/rkt",
    importpath = "github.com/google/cadvisor/manager/watcher/rkt",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/rkt/api/v1alpha:go_default_library",
        "//vendor/github.com/google/cadvisor/container/rkt:go_default_library",
        "//vendor/github.com/google/cadvisor/manager/watcher:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
