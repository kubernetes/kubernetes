load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "chain_provider.go",
        "context_background_go1.5.go",
        "context_background_go1.7.go",
        "context_go1.5.go",
        "context_go1.9.go",
        "credentials.go",
        "env_provider.go",
        "shared_credentials_provider.go",
        "static_provider.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/aws/aws-sdk-go/aws/credentials",
    importpath = "github.com/aws/aws-sdk-go/aws/credentials",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/aws/aws-sdk-go/aws/awserr:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/internal/context:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/internal/ini:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/internal/shareddefaults:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/internal/sync/singleflight:go_default_library",
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
        "//vendor/github.com/aws/aws-sdk-go/aws/credentials/processcreds:all-srcs",
        "//vendor/github.com/aws/aws-sdk-go/aws/credentials/stscreds:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
