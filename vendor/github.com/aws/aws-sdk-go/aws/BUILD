load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "config.go",
        "context_1_5.go",
        "context_1_9.go",
        "context_background_1_5.go",
        "context_background_1_7.go",
        "context_sleep.go",
        "convert_types.go",
        "doc.go",
        "errors.go",
        "jsonvalue.go",
        "logger.go",
        "types.go",
        "url.go",
        "url_1_7.go",
        "version.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/aws/aws-sdk-go/aws",
    importpath = "github.com/aws/aws-sdk-go/aws",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/aws/aws-sdk-go/aws/awserr:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/aws/credentials:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/aws/endpoints:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/internal/context:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/internal/sdkio:go_default_library",
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
        "//vendor/github.com/aws/aws-sdk-go/aws/csm:all-srcs",
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
