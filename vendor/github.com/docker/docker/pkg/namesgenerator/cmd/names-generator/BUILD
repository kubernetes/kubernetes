load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["main.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/pkg/namesgenerator/cmd/names-generator",
    importpath = "github.com/docker/docker/pkg/namesgenerator/cmd/names-generator",
    visibility = ["//visibility:private"],
    deps = ["//vendor/github.com/docker/docker/pkg/namesgenerator:go_default_library"],
)

go_binary(
    name = "names-generator",
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
