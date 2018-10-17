load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["main.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/gen",
    importpath = "github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/gen",
    visibility = ["//visibility:private"],
    deps = ["//vendor/github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/meta:go_default_library"],
)

go_binary(
    name = "gen",
    embed = [":go_default_library"],
    visibility = ["//visibility:public"],
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
