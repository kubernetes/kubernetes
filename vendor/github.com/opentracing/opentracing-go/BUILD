package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "globaltracer.go",
        "gocontext.go",
        "noop.go",
        "propagation.go",
        "span.go",
        "tracer.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/github.com/opentracing/opentracing-go/log:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
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
        "//vendor/github.com/opentracing/opentracing-go/ext:all-srcs",
        "//vendor/github.com/opentracing/opentracing-go/log:all-srcs",
    ],
    tags = ["automanaged"],
)
