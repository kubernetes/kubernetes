load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["node.go"],
    importpath = "gopkg.in/src-d/go-git.v4/utils/merkletrie/filesystem",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/gopkg.in/src-d/go-billy.v3:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/filemode:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/utils/merkletrie/noder:go_default_library",
    ],
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
