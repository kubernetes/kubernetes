load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["parsers.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/pkg/parsers",
    importpath = "github.com/docker/docker/pkg/parsers",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/docker/docker/pkg/parsers/operatingsystem:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
