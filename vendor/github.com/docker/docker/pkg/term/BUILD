load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "deprecated.go",
        "deprecated_unix.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/pkg/term",
    importpath = "github.com/docker/docker/pkg/term",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/moby/term:go_default_library"],
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
