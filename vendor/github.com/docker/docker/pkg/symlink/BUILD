load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "fs.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "fs_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "fs_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "fs_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "fs_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "fs_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "fs_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "fs_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "fs_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "fs_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "fs_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "fs_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/docker/docker/pkg/symlink",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/docker/docker/pkg/system:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/docker/docker/pkg/longpath:go_default_library",
            "//vendor/golang.org/x/sys/windows:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
