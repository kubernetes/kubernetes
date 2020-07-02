load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["utils.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/containernetworking/cni/pkg/utils",
    importpath = "github.com/containernetworking/cni/pkg/utils",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/containernetworking/cni/pkg/types:go_default_library"],
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
