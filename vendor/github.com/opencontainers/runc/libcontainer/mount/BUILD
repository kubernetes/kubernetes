load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "mount.go",
        "mountinfo.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
            "mount_linux.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/opencontainers/runc/libcontainer/mount",
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
