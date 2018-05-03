load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "common.go",
        "names.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "common_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "common_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "common_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "common_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "common_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "common_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "common_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "common_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "common_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "common_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "common_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/docker/docker/api",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/docker/docker/pkg/ioutils:go_default_library",
        "//vendor/github.com/docker/docker/pkg/system:go_default_library",
        "//vendor/github.com/docker/libtrust:go_default_library",
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
        "//vendor/github.com/docker/docker/api/types:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
