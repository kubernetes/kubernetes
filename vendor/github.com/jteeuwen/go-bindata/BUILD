load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "asset.go",
        "bytewriter.go",
        "config.go",
        "convert.go",
        "debug.go",
        "doc.go",
        "release.go",
        "restore.go",
        "stringwriter.go",
        "toc.go",
    ],
    visibility = ["//visibility:public"],
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
        "//vendor/github.com/jteeuwen/go-bindata/go-bindata:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
