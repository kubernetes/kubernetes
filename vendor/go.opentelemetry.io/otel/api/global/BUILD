load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["global.go"],
    importmap = "k8s.io/kubernetes/vendor/go.opentelemetry.io/otel/api/global",
    importpath = "go.opentelemetry.io/otel/api/global",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/go.opentelemetry.io/otel/api/global/internal:go_default_library",
        "//vendor/go.opentelemetry.io/otel/api/metric:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/go.opentelemetry.io/otel/api/global/internal:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
