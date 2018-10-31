load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "url.go",
        "url_unix.go",
        "url_windows.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/hashicorp/go-getter/helper/url",
    importpath = "github.com/hashicorp/go-getter/helper/url",
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
