load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["xlog.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/ulikunitz/xz/internal/xlog",
    importpath = "github.com/ulikunitz/xz/internal/xlog",
    visibility = ["//vendor/github.com/ulikunitz/xz:__subpackages__"],
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
