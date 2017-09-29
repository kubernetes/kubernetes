load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "auth.go",
        "client.go",
        "configs.go",
        "seccomp.go",
        "stats.go",
        "types.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/docker/engine-api/types/container:go_default_library",
        "//vendor/github.com/docker/engine-api/types/filters:go_default_library",
        "//vendor/github.com/docker/engine-api/types/network:go_default_library",
        "//vendor/github.com/docker/engine-api/types/registry:go_default_library",
        "//vendor/github.com/docker/go-connections/nat:go_default_library",
        "//vendor/github.com/docker/go-units:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/docker/engine-api/types/blkiodev:all-srcs",
        "//vendor/github.com/docker/engine-api/types/container:all-srcs",
        "//vendor/github.com/docker/engine-api/types/filters:all-srcs",
        "//vendor/github.com/docker/engine-api/types/network:all-srcs",
        "//vendor/github.com/docker/engine-api/types/reference:all-srcs",
        "//vendor/github.com/docker/engine-api/types/registry:all-srcs",
        "//vendor/github.com/docker/engine-api/types/strslice:all-srcs",
        "//vendor/github.com/docker/engine-api/types/time:all-srcs",
        "//vendor/github.com/docker/engine-api/types/versions:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
