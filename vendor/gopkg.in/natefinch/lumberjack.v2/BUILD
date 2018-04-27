load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "lumberjack.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "chown.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "chown.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "chown.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "chown.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "chown_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "chown.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "chown.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "chown.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "chown.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "chown.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "chown.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "gopkg.in/natefinch/lumberjack.v2",
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
