load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["cloudinfo.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/google/cadvisor/utils/cloudinfo",
    importpath = "github.com/google/cadvisor/utils/cloudinfo",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
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
        "//vendor/github.com/google/cadvisor/utils/cloudinfo/aws:all-srcs",
        "//vendor/github.com/google/cadvisor/utils/cloudinfo/azure:all-srcs",
        "//vendor/github.com/google/cadvisor/utils/cloudinfo/gce:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
