load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "common.go",
        "container.go",
        "network.go",
        "node.go",
        "secret.go",
        "service.go",
        "swarm.go",
        "task.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/docker/docker/api/types/container:go_default_library",
        "//vendor/github.com/docker/docker/api/types/mount:go_default_library",
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
