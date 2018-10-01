load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "aws_identity_doc_utils.go",
        "gcp_metadata_config.go",
        "monitored_resources.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/contrib.go.opencensus.io/exporter/stackdriver/monitoredresource",
    importpath = "contrib.go.opencensus.io/exporter/stackdriver/monitoredresource",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/cloud.google.com/go/compute/metadata:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/aws/ec2metadata:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/aws/session:go_default_library",
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
