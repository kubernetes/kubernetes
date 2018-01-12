load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "parser.go",
        "scanner.go",
        "token.go",
    ],
    importpath = "gopkg.in/src-d/go-git.v4/internal/revision",
    visibility = ["//vendor/gopkg.in/src-d/go-git.v4:__subpackages__"],
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
