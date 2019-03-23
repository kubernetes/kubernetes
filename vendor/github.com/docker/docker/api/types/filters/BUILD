load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["parse.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/api/types/filters",
    importpath = "github.com/docker/docker/api/types/filters",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/docker/docker/api/types/versions:go_default_library"],
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
