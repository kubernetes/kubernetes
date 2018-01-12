load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["storer.go"],
    importpath = "gopkg.in/src-d/go-git.v4/storage",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/gopkg.in/src-d/go-git.v4/config:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/storer:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/storage/filesystem:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/storage/memory:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
