load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["oomparser.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/google/cadvisor/utils/oomparser",
    importpath = "github.com/google/cadvisor/utils/oomparser",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/euank/go-kmsg-parser/kmsgparser:go_default_library",
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
