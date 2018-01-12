load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "index.go",
        "object.go",
        "reference.go",
        "shallow.go",
        "storer.go",
    ],
    importpath = "gopkg.in/src-d/go-git.v4/plumbing/storer",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/format/index:go_default_library",
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
