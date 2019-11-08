load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "post_go18.go",
        "pre_go18.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/go-openapi/analysis/internal",
    importpath = "github.com/go-openapi/analysis/internal",
    visibility = ["//vendor/github.com/go-openapi/analysis:__subpackages__"],
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
