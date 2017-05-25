package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "transport.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor/github.com/uber/jaeger-client-go/thrift-gen/zipkincore:go_default_library"],
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
        "//vendor/github.com/uber/jaeger-client-go/transport/udp:all-srcs",
    ],
    tags = ["automanaged"],
)
