load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["cpuload.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/google/cadvisor/utils/cpuload",
    importpath = "github.com/google/cadvisor/utils/cpuload",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
        "//vendor/github.com/google/cadvisor/utils/cpuload/netlink:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/google/cadvisor/utils/cpuload/netlink:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
