load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "exemplar.go",
        "label.go",
        "metric.go",
        "point.go",
        "type_string.go",
        "unit.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.opencensus.io/metric/metricdata",
    importpath = "go.opencensus.io/metric/metricdata",
    visibility = ["//visibility:public"],
    deps = ["//vendor/go.opencensus.io/resource:go_default_library"],
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
