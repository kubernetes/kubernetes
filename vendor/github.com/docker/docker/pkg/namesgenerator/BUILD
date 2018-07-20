load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["names-generator.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/pkg/namesgenerator",
    importpath = "github.com/docker/docker/pkg/namesgenerator",
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
        "//vendor/github.com/docker/docker/pkg/namesgenerator/cmd/names-generator:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
