package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = ["aws_credentials.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/credentialprovider:go_default_library",
        "//vendor:github.com/aws/aws-sdk-go/aws",
        "//vendor:github.com/aws/aws-sdk-go/aws/request",
        "//vendor:github.com/aws/aws-sdk-go/aws/session",
        "//vendor:github.com/aws/aws-sdk-go/service/ecr",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["aws_credentials_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/credentialprovider:go_default_library",
        "//vendor:github.com/aws/aws-sdk-go/aws",
        "//vendor:github.com/aws/aws-sdk-go/service/ecr",
    ],
)
