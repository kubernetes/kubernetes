load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "lru.go",
        "lru_interface.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/hashicorp/golang-lru/simplelru",
    importpath = "github.com/hashicorp/golang-lru/simplelru",
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
