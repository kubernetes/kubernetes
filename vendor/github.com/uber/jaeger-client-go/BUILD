package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "constants.go",
        "context.go",
        "doc.go",
        "interop.go",
        "jaeger_span.go",
        "jaeger_tag.go",
        "logger.go",
        "metrics.go",
        "observer.go",
        "propagation.go",
        "reference.go",
        "reporter.go",
        "reporter_options.go",
        "sampler.go",
        "sampler_options.go",
        "span.go",
        "thrift_span.go",
        "tracer.go",
        "tracer_options.go",
        "zipkin.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/github.com/opentracing/opentracing-go:go_default_library",
        "//vendor/github.com/opentracing/opentracing-go/ext:go_default_library",
        "//vendor/github.com/opentracing/opentracing-go/log:go_default_library",
        "//vendor/github.com/uber/jaeger-client-go/internal/spanlog:go_default_library",
        "//vendor/github.com/uber/jaeger-client-go/log:go_default_library",
        "//vendor/github.com/uber/jaeger-client-go/thrift-gen/jaeger:go_default_library",
        "//vendor/github.com/uber/jaeger-client-go/thrift-gen/sampling:go_default_library",
        "//vendor/github.com/uber/jaeger-client-go/thrift-gen/zipkincore:go_default_library",
        "//vendor/github.com/uber/jaeger-client-go/transport:go_default_library",
        "//vendor/github.com/uber/jaeger-client-go/utils:go_default_library",
        "//vendor/github.com/uber/jaeger-lib/metrics:go_default_library",
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
        "//vendor/github.com/uber/jaeger-client-go/config:all-srcs",
        "//vendor/github.com/uber/jaeger-client-go/internal/spanlog:all-srcs",
        "//vendor/github.com/uber/jaeger-client-go/log:all-srcs",
        "//vendor/github.com/uber/jaeger-client-go/rpcmetrics:all-srcs",
        "//vendor/github.com/uber/jaeger-client-go/thrift-gen/agent:all-srcs",
        "//vendor/github.com/uber/jaeger-client-go/thrift-gen/jaeger:all-srcs",
        "//vendor/github.com/uber/jaeger-client-go/thrift-gen/sampling:all-srcs",
        "//vendor/github.com/uber/jaeger-client-go/thrift-gen/zipkincore:all-srcs",
        "//vendor/github.com/uber/jaeger-client-go/transport:all-srcs",
        "//vendor/github.com/uber/jaeger-client-go/utils:all-srcs",
    ],
    tags = ["automanaged"],
)
