load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "big.go",
        "bigbytes.go",
        "bytes.go",
        "comma.go",
        "commaf.go",
        "ftoa.go",
        "humanize.go",
        "number.go",
        "ordinals.go",
        "si.go",
        "times.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/dustin/go-humanize",
    importpath = "github.com/dustin/go-humanize",
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
