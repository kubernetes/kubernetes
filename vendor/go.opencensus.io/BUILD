load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["opencensus.go"],
    importmap = "k8s.io/kubernetes/vendor/go.opencensus.io",
    importpath = "go.opencensus.io",
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
        "//vendor/go.opencensus.io/internal:all-srcs",
        "//vendor/go.opencensus.io/metric/metricdata:all-srcs",
        "//vendor/go.opencensus.io/metric/metricproducer:all-srcs",
        "//vendor/go.opencensus.io/plugin/ochttp:all-srcs",
        "//vendor/go.opencensus.io/resource:all-srcs",
        "//vendor/go.opencensus.io/stats:all-srcs",
        "//vendor/go.opencensus.io/tag:all-srcs",
        "//vendor/go.opencensus.io/trace:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
