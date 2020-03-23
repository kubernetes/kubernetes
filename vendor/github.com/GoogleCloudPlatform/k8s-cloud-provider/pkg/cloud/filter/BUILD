load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["filter.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/filter",
    importpath = "github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/filter",
    visibility = ["//visibility:public"],
    deps = ["//vendor/k8s.io/klog:go_default_library"],
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
