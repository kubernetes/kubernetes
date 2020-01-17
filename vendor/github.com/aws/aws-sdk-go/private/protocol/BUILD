load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "host.go",
        "host_prefix.go",
        "idempotency.go",
        "jsonvalue.go",
        "payload.go",
        "protocol.go",
        "timestamp.go",
        "unmarshal.go",
        "unmarshal_error.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/aws/aws-sdk-go/private/protocol",
    importpath = "github.com/aws/aws-sdk-go/private/protocol",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/aws/aws-sdk-go/aws:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/aws/awserr:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/aws/client/metadata:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/aws/request:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/internal/sdkmath:go_default_library",
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
        "//vendor/github.com/aws/aws-sdk-go/private/protocol/ec2query:all-srcs",
        "//vendor/github.com/aws/aws-sdk-go/private/protocol/json/jsonutil:all-srcs",
        "//vendor/github.com/aws/aws-sdk-go/private/protocol/jsonrpc:all-srcs",
        "//vendor/github.com/aws/aws-sdk-go/private/protocol/query:all-srcs",
        "//vendor/github.com/aws/aws-sdk-go/private/protocol/rest:all-srcs",
        "//vendor/github.com/aws/aws-sdk-go/private/protocol/xml/xmlutil:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
