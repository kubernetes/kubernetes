load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "dictionary.go",
        "read.go",
        "types.go",
        "write.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/spdystream/spdy",
    importpath = "github.com/docker/spdystream/spdy",
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
