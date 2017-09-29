load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "chain_provider.go",
        "credentials.go",
        "env_provider.go",
        "shared_credentials_provider.go",
        "static_provider.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/aws/aws-sdk-go/aws/awserr:go_default_library",
        "//vendor/github.com/go-ini/ini:go_default_library",
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
        "//vendor/github.com/aws/aws-sdk-go/aws/credentials/ec2rolecreds:all-srcs",
        "//vendor/github.com/aws/aws-sdk-go/aws/credentials/endpointcreds:all-srcs",
        "//vendor/github.com/aws/aws-sdk-go/aws/credentials/stscreds:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
