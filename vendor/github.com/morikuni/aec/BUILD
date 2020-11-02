load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "aec.go",
        "ansi.go",
        "builder.go",
        "sgr.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/morikuni/aec",
    importpath = "github.com/morikuni/aec",
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
