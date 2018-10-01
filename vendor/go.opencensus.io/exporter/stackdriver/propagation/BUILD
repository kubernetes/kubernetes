load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["http.go"],
    importmap = "k8s.io/kubernetes/vendor/go.opencensus.io/exporter/stackdriver/propagation",
    importpath = "go.opencensus.io/exporter/stackdriver/propagation",
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
