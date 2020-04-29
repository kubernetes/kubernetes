load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "api.go",
        "common.go",
        "counter.go",
        "doc.go",
        "kind_string.go",
        "measure.go",
        "must.go",
        "noop.go",
        "observer.go",
        "sdkhelpers.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.opentelemetry.io/otel/api/metric",
    importpath = "go.opentelemetry.io/otel/api/metric",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/go.opentelemetry.io/otel/api/core:go_default_library",
        "//vendor/go.opentelemetry.io/otel/api/unit:go_default_library",
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
        "//vendor/go.opentelemetry.io/otel/api/metric/registry:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
