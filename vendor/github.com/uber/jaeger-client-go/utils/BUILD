package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "http_json.go",
        "localip.go",
        "rand.go",
        "rate_limiter.go",
        "udp_client.go",
        "utils.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/github.com/apache/thrift/lib/go/thrift:go_default_library",
        "//vendor/github.com/uber/jaeger-client-go/thrift-gen/agent:go_default_library",
        "//vendor/github.com/uber/jaeger-client-go/thrift-gen/zipkincore:go_default_library",
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
)
