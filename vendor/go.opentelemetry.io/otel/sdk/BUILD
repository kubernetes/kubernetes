load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["opentelemetry.go"],
    importmap = "k8s.io/kubernetes/vendor/go.opentelemetry.io/otel/sdk",
    importpath = "go.opentelemetry.io/otel/sdk",
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
        "//vendor/go.opentelemetry.io/otel/sdk/export/trace:all-srcs",
        "//vendor/go.opentelemetry.io/otel/sdk/internal:all-srcs",
        "//vendor/go.opentelemetry.io/otel/sdk/resource:all-srcs",
        "//vendor/go.opentelemetry.io/otel/sdk/trace:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
