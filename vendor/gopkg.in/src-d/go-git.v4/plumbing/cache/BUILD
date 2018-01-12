load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "common.go",
        "object_lru.go",
        "queue.go",
    ],
    importpath = "gopkg.in/src-d/go-git.v4/plumbing/cache",
    visibility = ["//visibility:public"],
    deps = ["//vendor/gopkg.in/src-d/go-git.v4/plumbing:go_default_library"],
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
