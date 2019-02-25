load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "jsonlog.go",
        "jsonlogbytes.go",
        "time_marshalling.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/daemon/logger/jsonfilelog/jsonlog",
    importpath = "github.com/docker/docker/daemon/logger/jsonfilelog/jsonlog",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/pkg/errors:go_default_library"],
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
