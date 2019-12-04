load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "manager.go",
        "producer.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.opencensus.io/metric/metricproducer",
    importpath = "go.opencensus.io/metric/metricproducer",
    visibility = ["//visibility:public"],
    deps = ["//vendor/go.opencensus.io/metric/metricdata:go_default_library"],
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
