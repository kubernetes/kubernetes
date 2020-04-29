load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "meter.go",
        "state.go",
        "trace.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.opentelemetry.io/otel/api/global/internal",
    importpath = "go.opentelemetry.io/otel/api/global/internal",
    visibility = ["//vendor/go.opentelemetry.io/otel/api/global:__subpackages__"],
    deps = [
        "//vendor/go.opentelemetry.io/otel/api/core:go_default_library",
        "//vendor/go.opentelemetry.io/otel/api/correlation:go_default_library",
        "//vendor/go.opentelemetry.io/otel/api/metric:go_default_library",
        "//vendor/go.opentelemetry.io/otel/api/metric/registry:go_default_library",
        "//vendor/go.opentelemetry.io/otel/api/propagation:go_default_library",
        "//vendor/go.opentelemetry.io/otel/api/trace:go_default_library",
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
