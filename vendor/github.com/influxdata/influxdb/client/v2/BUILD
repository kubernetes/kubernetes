load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "client.go",
        "udp.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/influxdata/influxdb/client/v2",
    importpath = "github.com/influxdata/influxdb/client/v2",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/influxdata/influxdb/models:go_default_library"],
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
