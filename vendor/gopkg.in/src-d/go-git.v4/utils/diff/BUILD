load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["diff.go"],
    importpath = "gopkg.in/src-d/go-git.v4/utils/diff",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/sergi/go-diff/diffmatchpatch:go_default_library"],
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
