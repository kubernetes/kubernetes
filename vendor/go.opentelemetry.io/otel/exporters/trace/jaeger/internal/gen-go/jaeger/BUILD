load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "GoUnusedProtection__.go",
        "agent.go",
        "jaeger.go",
        "jaeger-consts.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.opentelemetry.io/otel/exporters/trace/jaeger/internal/gen-go/jaeger",
    importpath = "go.opentelemetry.io/otel/exporters/trace/jaeger/internal/gen-go/jaeger",
    visibility = ["//vendor/go.opentelemetry.io/otel/exporters/trace/jaeger:__subpackages__"],
    deps = ["//vendor/github.com/apache/thrift/lib/go/thrift:go_default_library"],
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
