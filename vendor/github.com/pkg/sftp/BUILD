load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "attrs.go",
        "client.go",
        "conn.go",
        "packet.go",
        "release.go",
        "server.go",
        "sftp.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "attrs_stubs.go",
            "server_statvfs_stubs.go",
            "server_stubs.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "attrs_stubs.go",
            "attrs_unix.go",
            "server_statvfs_darwin.go",
            "server_statvfs_impl.go",
            "server_stubs.go",
            "server_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "attrs_stubs.go",
            "attrs_unix.go",
            "server_statvfs_stubs.go",
            "server_stubs.go",
            "server_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "attrs_stubs.go",
            "attrs_unix.go",
            "server_statvfs_stubs.go",
            "server_stubs.go",
            "server_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "attrs_stubs.go",
            "attrs_unix.go",
            "server_statvfs_impl.go",
            "server_statvfs_linux.go",
            "server_stubs.go",
            "server_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "attrs_stubs.go",
            "server_statvfs_stubs.go",
            "server_stubs.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "attrs_stubs.go",
            "attrs_unix.go",
            "server_statvfs_stubs.go",
            "server_stubs.go",
            "server_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "attrs_stubs.go",
            "attrs_unix.go",
            "server_statvfs_stubs.go",
            "server_stubs.go",
            "server_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "server_statvfs_stubs.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "attrs_stubs.go",
            "attrs_unix.go",
            "server_statvfs_stubs.go",
            "server_stubs.go",
            "server_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "attrs_stubs.go",
            "server_statvfs_stubs.go",
            "server_stubs.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/pkg/sftp",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/kr/fs:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/golang.org/x/crypto/ssh:go_default_library",
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
