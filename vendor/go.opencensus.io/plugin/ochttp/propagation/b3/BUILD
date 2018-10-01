load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["b3.go"],
    importmap = "k8s.io/kubernetes/vendor/go.opencensus.io/plugin/ochttp/propagation/b3",
    importpath = "go.opencensus.io/plugin/ochttp/propagation/b3",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/go.opencensus.io/trace:go_default_library",
        "//vendor/go.opencensus.io/trace/propagation:go_default_library",
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
