load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "debug.go",
        "decode.go",
        "encode.go",
        "error.go",
        "header.go",
        "header_value.go",
        "message.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/aws/aws-sdk-go/private/protocol/eventstream",
    importpath = "github.com/aws/aws-sdk-go/private/protocol/eventstream",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/aws/aws-sdk-go/aws:go_default_library"],
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
        "//vendor/github.com/aws/aws-sdk-go/private/protocol/eventstream/eventstreamapi:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
