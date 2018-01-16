load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = select({
        "@io_bazel_rules_go//go/platform:android": [
            "cgroups_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "cgroups_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "cgroups_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "cgroups_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "cgroups.go",
            "stats.go",
            "utils.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "cgroups_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "cgroups_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "cgroups_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "cgroups_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "cgroups_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "cgroups_unsupported.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/opencontainers/runc/libcontainer/cgroups",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/docker/go-units:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
        ],
        "//conditions:default": [],
    }),
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
        "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/fs:all-srcs",
        "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/systemd:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
