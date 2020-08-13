load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "error.go",
        "utils.go",
        "utils_notsolaris.go",
        "utils_solaris.go",
        "zfs.go",
        "zpool.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/mistifyio/go-zfs",
    importpath = "github.com/mistifyio/go-zfs",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/google/uuid:go_default_library"],
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
