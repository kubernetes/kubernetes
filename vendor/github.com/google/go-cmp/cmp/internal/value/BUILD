load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "pointer_unsafe.go",
        "sort.go",
        "zero.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/google/go-cmp/cmp/internal/value",
    importpath = "github.com/google/go-cmp/cmp/internal/value",
    visibility = ["//vendor/github.com/google/go-cmp/cmp:__subpackages__"],
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
