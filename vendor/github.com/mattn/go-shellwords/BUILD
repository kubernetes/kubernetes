load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "shellwords.go",
        "util_go15.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "util_posix.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "util_posix.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "util_posix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "util_posix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "util_posix.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "util_posix.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "util_posix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "util_posix.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "util_posix.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "util_posix.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "util_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/mattn/go-shellwords",
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
