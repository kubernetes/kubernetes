load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "systemstat.go",
        "systemstat_ex.go",
        "systemstat_linux.go",
        "utils.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/bitbucket.org/bertimus9/systemstat",
    importpath = "bitbucket.org/bertimus9/systemstat",
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
