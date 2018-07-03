load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "build.go",
        "unmarshal.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/aws/aws-sdk-go/private/protocol/ec2query",
    importpath = "github.com/aws/aws-sdk-go/private/protocol/ec2query",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/aws/aws-sdk-go/aws/awserr:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/aws/request:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/private/protocol/query/queryutil:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/private/protocol/xml/xmlutil:go_default_library",
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
