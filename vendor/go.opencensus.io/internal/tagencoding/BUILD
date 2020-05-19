load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["tagencoding.go"],
    importmap = "k8s.io/kubernetes/vendor/go.opencensus.io/internal/tagencoding",
    importpath = "go.opencensus.io/internal/tagencoding",
    visibility = ["//vendor/go.opencensus.io:__subpackages__"],
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
