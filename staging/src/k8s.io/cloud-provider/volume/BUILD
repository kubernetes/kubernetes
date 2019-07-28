load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["constants.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/cloud-provider/volume",
    importpath = "k8s.io/cloud-provider/volume",
    visibility = ["//visibility:public"],
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
        "//staging/src/k8s.io/cloud-provider/volume/errors:all-srcs",
        "//staging/src/k8s.io/cloud-provider/volume/helpers:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
