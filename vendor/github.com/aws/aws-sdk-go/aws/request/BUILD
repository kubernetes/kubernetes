load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "handlers.go",
        "http_request.go",
        "offset_reader.go",
        "request.go",
        "request_1_7.go",
        "request_1_8.go",
        "request_context.go",
        "request_context_1_6.go",
        "request_pagination.go",
        "retryer.go",
        "timeout_read_closer.go",
        "validation.go",
        "waiter.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "connection_reset_error.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "connection_reset_error.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "connection_reset_error.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "connection_reset_error.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "connection_reset_error.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "connection_reset_error.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "connection_reset_error.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "connection_reset_error.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "connection_reset_error_other.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "connection_reset_error.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "connection_reset_error.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/aws/aws-sdk-go/aws/request",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/aws/aws-sdk-go/aws:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/aws/awserr:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/aws/awsutil:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/aws/client/metadata:go_default_library",
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
