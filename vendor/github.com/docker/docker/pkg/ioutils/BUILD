load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "buffer.go",
        "bytespipe.go",
        "fswriters.go",
        "readers.go",
        "writeflusher.go",
        "writers.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "temp_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "temp_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "temp_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "temp_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "temp_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "temp_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "temp_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "temp_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "temp_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "temp_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "temp_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/docker/docker/pkg/ioutils",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/net/context:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/docker/docker/pkg/longpath:go_default_library",
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
