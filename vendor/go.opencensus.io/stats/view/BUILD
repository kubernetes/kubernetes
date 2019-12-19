load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "aggregation.go",
        "aggregation_data.go",
        "collector.go",
        "doc.go",
        "export.go",
        "view.go",
        "view_to_metric.go",
        "worker.go",
        "worker_commands.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.opencensus.io/stats/view",
    importpath = "go.opencensus.io/stats/view",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/go.opencensus.io/internal/tagencoding:go_default_library",
        "//vendor/go.opencensus.io/metric/metricdata:go_default_library",
        "//vendor/go.opencensus.io/metric/metricproducer:go_default_library",
        "//vendor/go.opencensus.io/stats:go_default_library",
        "//vendor/go.opencensus.io/stats/internal:go_default_library",
        "//vendor/go.opencensus.io/tag:go_default_library",
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
