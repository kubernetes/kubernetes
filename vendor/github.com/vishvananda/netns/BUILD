load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "netns.go",
        "netns_linux.go",
        "netns_unspecified.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/vishvananda/netns",
    importpath = "github.com/vishvananda/netns",
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
