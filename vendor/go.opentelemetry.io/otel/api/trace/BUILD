load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "always_off_sampler.go",
        "always_on_sampler.go",
        "api.go",
        "b3_propagator.go",
        "context.go",
        "doc.go",
        "noop_span.go",
        "noop_trace.go",
        "noop_trace_provider.go",
        "sampler.go",
        "trace_context_propagator.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.opentelemetry.io/otel/api/trace",
    importpath = "go.opentelemetry.io/otel/api/trace",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/go.opentelemetry.io/otel/api/core:go_default_library",
        "//vendor/go.opentelemetry.io/otel/api/propagation:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
