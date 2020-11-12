load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "data_source.go",
        "deprecated.go",
        "error.go",
        "file.go",
        "helper.go",
        "ini.go",
        "key.go",
        "parser.go",
        "section.go",
        "struct.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gopkg.in/ini.v1",
    importpath = "gopkg.in/ini.v1",
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
