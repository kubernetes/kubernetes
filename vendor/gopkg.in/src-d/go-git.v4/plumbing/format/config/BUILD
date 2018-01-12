load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "common.go",
        "decoder.go",
        "doc.go",
        "encoder.go",
        "option.go",
        "section.go",
    ],
    importpath = "gopkg.in/src-d/go-git.v4/plumbing/format/config",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/src-d/gcfg:go_default_library"],
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
