load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bitset.go",
        "popcnt.go",
        "popcnt_19.go",
        "popcnt_amd64.go",
        "popcnt_amd64.s",
        "popcnt_generic.go",
        "trailing_zeros_18.go",
        "trailing_zeros_19.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/willf/bitset",
    importpath = "github.com/willf/bitset",
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
