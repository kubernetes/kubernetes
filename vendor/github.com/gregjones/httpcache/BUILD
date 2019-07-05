load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["httpcache.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/gregjones/httpcache",
    importpath = "github.com/gregjones/httpcache",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/gregjones/httpcache/diskcache:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
