load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["frame.go"],
    importpath = "gopkg.in/src-d/go-git.v4/utils/merkletrie/internal/frame",
    visibility = ["//vendor/gopkg.in/src-d/go-git.v4/utils/merkletrie:__subpackages__"],
    deps = ["//vendor/gopkg.in/src-d/go-git.v4/utils/merkletrie/noder:go_default_library"],
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
