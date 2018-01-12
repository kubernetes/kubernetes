load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "errors.go",
        "go1_0.go",
        "go1_2.go",
        "read.go",
        "set.go",
    ],
    importpath = "github.com/src-d/gcfg",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/src-d/gcfg/scanner:go_default_library",
        "//vendor/github.com/src-d/gcfg/token:go_default_library",
        "//vendor/github.com/src-d/gcfg/types:go_default_library",
        "//vendor/gopkg.in/warnings.v0:go_default_library",
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
        "//vendor/github.com/src-d/gcfg/scanner:all-srcs",
        "//vendor/github.com/src-d/gcfg/token:all-srcs",
        "//vendor/github.com/src-d/gcfg/types:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
