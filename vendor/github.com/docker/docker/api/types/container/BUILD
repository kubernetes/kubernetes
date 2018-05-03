load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "config.go",
        "container_changes.go",
        "container_create.go",
        "container_top.go",
        "container_update.go",
        "container_wait.go",
        "host_config.go",
        "waitcondition.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "hostconfig_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "hostconfig_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "hostconfig_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "hostconfig_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "hostconfig_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "hostconfig_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "hostconfig_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "hostconfig_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "hostconfig_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "hostconfig_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "hostconfig_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/docker/docker/api/types/container",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/docker/docker/api/types/blkiodev:go_default_library",
        "//vendor/github.com/docker/docker/api/types/mount:go_default_library",
        "//vendor/github.com/docker/docker/api/types/strslice:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
