load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "shellwords.go",
        "util_go15.go",
        "util_posix.go",
        "util_windows.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/mattn/go-shellwords",
    importpath = "github.com/mattn/go-shellwords",
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
