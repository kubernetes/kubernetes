load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "measure.go",
        "measure_float64.go",
        "measure_int64.go",
        "record.go",
        "units.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.opencensus.io/stats",
    importpath = "go.opencensus.io/stats",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/go.opencensus.io/metric/metricdata:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/go.opencensus.io/stats/internal:all-srcs",
        "//vendor/go.opencensus.io/stats/view:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
