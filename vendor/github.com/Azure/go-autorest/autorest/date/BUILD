load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "date.go",
        "time.go",
        "timerfc1123.go",
        "unixtime.go",
        "utility.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/Azure/go-autorest/autorest/date",
    importpath = "github.com/Azure/go-autorest/autorest/date",
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
