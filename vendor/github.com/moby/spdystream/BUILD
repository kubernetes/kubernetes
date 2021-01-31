load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "connection.go",
        "handlers.go",
        "priority.go",
        "stream.go",
        "utils.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/moby/spdystream",
    importpath = "github.com/moby/spdystream",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/moby/spdystream/spdy:go_default_library"],
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
        "//vendor/github.com/moby/spdystream/spdy:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
