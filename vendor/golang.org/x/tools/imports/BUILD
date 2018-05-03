load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "fastwalk.go",
        "fix.go",
        "imports.go",
        "sortimports.go",
        "zstdlib.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "fastwalk_portable.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "fastwalk_dirent_ino.go",
            "fastwalk_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "fastwalk_portable.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "fastwalk_dirent_fileno.go",
            "fastwalk_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "fastwalk_dirent_ino.go",
            "fastwalk_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "fastwalk_portable.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "fastwalk_dirent_fileno.go",
            "fastwalk_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "fastwalk_dirent_fileno.go",
            "fastwalk_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "fastwalk_portable.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "fastwalk_portable.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "fastwalk_portable.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "golang.org/x/tools/imports",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/tools/go/ast/astutil:go_default_library",
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
