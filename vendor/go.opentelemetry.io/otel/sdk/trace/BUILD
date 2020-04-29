load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "attributesMap.go",
        "batch_span_processor.go",
        "config.go",
        "doc.go",
        "evictedqueue.go",
        "id_generator.go",
        "provider.go",
        "sampling.go",
        "simple_span_processor.go",
        "span.go",
        "span_processor.go",
        "trace.go",
        "trace_go11.go",
        "trace_nongo11.go",
        "tracer.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.opentelemetry.io/otel/sdk/trace",
    importpath = "go.opentelemetry.io/otel/sdk/trace",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/go.opentelemetry.io/otel/api/core:go_default_library",
        "//vendor/go.opentelemetry.io/otel/api/trace:go_default_library",
        "//vendor/go.opentelemetry.io/otel/internal/trace/parent:go_default_library",
        "//vendor/go.opentelemetry.io/otel/sdk/export/trace:go_default_library",
        "//vendor/go.opentelemetry.io/otel/sdk/internal:go_default_library",
        "//vendor/go.opentelemetry.io/otel/sdk/resource:go_default_library",
        "//vendor/go.opentelemetry.io/otel/sdk/trace/internal:go_default_library",
        "//vendor/google.golang.org/grpc/codes:go_default_library",
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
        "//vendor/go.opentelemetry.io/otel/sdk/trace/internal:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
