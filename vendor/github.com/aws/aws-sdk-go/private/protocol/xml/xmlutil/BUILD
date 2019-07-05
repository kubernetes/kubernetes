load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "build.go",
        "unmarshal.go",
        "xml_to_struct.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/aws/aws-sdk-go/private/protocol/xml/xmlutil",
    importpath = "github.com/aws/aws-sdk-go/private/protocol/xml/xmlutil",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/aws/aws-sdk-go/private/protocol:go_default_library"],
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
