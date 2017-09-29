load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "config.go",
        "convert_types.go",
        "errors.go",
        "logger.go",
        "types.go",
        "version.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/aws/aws-sdk-go/aws/awserr:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/aws/credentials:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/aws/endpoints:go_default_library",
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
        "//vendor/github.com/aws/aws-sdk-go/aws/awserr:all-srcs",
        "//vendor/github.com/aws/aws-sdk-go/aws/awsutil:all-srcs",
        "//vendor/github.com/aws/aws-sdk-go/aws/client:all-srcs",
        "//vendor/github.com/aws/aws-sdk-go/aws/corehandlers:all-srcs",
        "//vendor/github.com/aws/aws-sdk-go/aws/credentials:all-srcs",
        "//vendor/github.com/aws/aws-sdk-go/aws/defaults:all-srcs",
        "//vendor/github.com/aws/aws-sdk-go/aws/ec2metadata:all-srcs",
        "//vendor/github.com/aws/aws-sdk-go/aws/endpoints:all-srcs",
        "//vendor/github.com/aws/aws-sdk-go/aws/request:all-srcs",
        "//vendor/github.com/aws/aws-sdk-go/aws/session:all-srcs",
        "//vendor/github.com/aws/aws-sdk-go/aws/signer/v4:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
