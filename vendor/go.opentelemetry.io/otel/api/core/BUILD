load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "key.go",
        "number.go",
        "numberkind_string.go",
        "rawhelpers.go",
        "span_context.go",
        "valuetype_string.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.opentelemetry.io/otel/api/core",
    importpath = "go.opentelemetry.io/otel/api/core",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
