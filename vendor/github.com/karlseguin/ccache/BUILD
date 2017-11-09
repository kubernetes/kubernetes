load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bucket.go",
        "cache.go",
        "configuration.go",
        "item.go",
        "layeredbucket.go",
        "layeredcache.go",
        "secondarycache.go",
    ],
    importpath = "github.com/karlseguin/ccache",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
