load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["gce.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/cloud-provider/features",
    importpath = "k8s.io/cloud-provider/features",
    visibility = ["//visibility:public"],
    deps = ["//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library"],
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
