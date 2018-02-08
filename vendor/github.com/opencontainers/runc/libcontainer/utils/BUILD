load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "utils.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "utils_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "utils_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "utils_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "utils_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "cmsg.go",
            "utils_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "utils_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "utils_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "utils_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "utils_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "utils_unix.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/opencontainers/runc/libcontainer/utils",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/sys/unix:go_default_library",
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
