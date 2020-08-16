load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "xxhash.go",
        "xxhash_amd64.go",
        "xxhash_amd64.s",
        "xxhash_other.go",
        "xxhash_unsafe.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/cespare/xxhash/v2",
    importpath = "github.com/cespare/xxhash/v2",
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
