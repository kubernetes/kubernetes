load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["testutils.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/daemon/events/testutils",
    importpath = "github.com/docker/docker/daemon/events/testutils",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/docker/docker/api/types/events:go_default_library",
        "//vendor/github.com/docker/docker/api/types/time:go_default_library",
    ],
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
