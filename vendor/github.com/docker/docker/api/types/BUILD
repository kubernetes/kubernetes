load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "auth.go",
        "client.go",
        "configs.go",
        "error_response.go",
        "id_response.go",
        "image_summary.go",
        "plugin.go",
        "plugin_device.go",
        "plugin_env.go",
        "plugin_interface_type.go",
        "plugin_mount.go",
        "plugin_responses.go",
        "port.go",
        "seccomp.go",
        "service_update_response.go",
        "stats.go",
        "types.go",
        "volume.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/docker/docker/api/types/container:go_default_library",
        "//vendor/github.com/docker/docker/api/types/filters:go_default_library",
        "//vendor/github.com/docker/docker/api/types/mount:go_default_library",
        "//vendor/github.com/docker/docker/api/types/network:go_default_library",
        "//vendor/github.com/docker/docker/api/types/registry:go_default_library",
        "//vendor/github.com/docker/docker/api/types/swarm:go_default_library",
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
        "//vendor/github.com/docker/docker/api/types/blkiodev:all-srcs",
        "//vendor/github.com/docker/docker/api/types/container:all-srcs",
        "//vendor/github.com/docker/docker/api/types/events:all-srcs",
        "//vendor/github.com/docker/docker/api/types/filters:all-srcs",
        "//vendor/github.com/docker/docker/api/types/mount:all-srcs",
        "//vendor/github.com/docker/docker/api/types/network:all-srcs",
        "//vendor/github.com/docker/docker/api/types/reference:all-srcs",
        "//vendor/github.com/docker/docker/api/types/registry:all-srcs",
        "//vendor/github.com/docker/docker/api/types/strslice:all-srcs",
        "//vendor/github.com/docker/docker/api/types/swarm:all-srcs",
        "//vendor/github.com/docker/docker/api/types/time:all-srcs",
        "//vendor/github.com/docker/docker/api/types/versions:all-srcs",
        "//vendor/github.com/docker/docker/api/types/volume:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
