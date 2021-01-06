load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "asset.go",
        "bits.go",
        "bytewriter.go",
        "config.go",
        "convert.go",
        "debug.go",
        "doc.go",
        "filewriter.go",
        "release.go",
        "release_go112.go",
        "release_go113.go",
        "restore.go",
        "safefile.go",
        "stringwriter.go",
        "toc.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/kevinburke/go-bindata",
    importpath = "github.com/kevinburke/go-bindata",
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
        "//vendor/github.com/kevinburke/go-bindata/go-bindata:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
