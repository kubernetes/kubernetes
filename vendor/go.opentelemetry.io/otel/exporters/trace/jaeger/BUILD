load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "agent.go",
        "doc.go",
        "jaeger.go",
        "uploader.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.opentelemetry.io/otel/exporters/trace/jaeger",
    importpath = "go.opentelemetry.io/otel/exporters/trace/jaeger",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/apache/thrift/lib/go/thrift:go_default_library",
        "//vendor/go.opentelemetry.io/otel/api/core:go_default_library",
        "//vendor/go.opentelemetry.io/otel/api/global:go_default_library",
        "//vendor/go.opentelemetry.io/otel/exporters/trace/jaeger/internal/gen-go/jaeger:go_default_library",
        "//vendor/go.opentelemetry.io/otel/sdk/export/trace:go_default_library",
        "//vendor/go.opentelemetry.io/otel/sdk/trace:go_default_library",
        "//vendor/google.golang.org/api/support/bundler:go_default_library",
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
        "//vendor/go.opentelemetry.io/otel/exporters/trace/jaeger/internal/gen-go/jaeger:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
