load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["record.go"],
    importmap = "k8s.io/kubernetes/vendor/go.opencensus.io/stats/internal",
    importpath = "go.opencensus.io/stats/internal",
    visibility = ["//vendor/go.opencensus.io/stats:__subpackages__"],
    deps = ["//vendor/go.opencensus.io/tag:go_default_library"],
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
