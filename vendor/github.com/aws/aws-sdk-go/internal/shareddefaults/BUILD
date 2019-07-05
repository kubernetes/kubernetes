load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "ecs_container.go",
        "shared_config.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/aws/aws-sdk-go/internal/shareddefaults",
    importpath = "github.com/aws/aws-sdk-go/internal/shareddefaults",
    visibility = ["//vendor/github.com/aws/aws-sdk-go:__subpackages__"],
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
