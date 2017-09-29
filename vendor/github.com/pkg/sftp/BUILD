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
        "server_statvfs_stubs.go",
        "sftp.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "attrs_unix.go",
            "server_statvfs_darwin.go",
            "server_statvfs_impl.go",
            "server_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "attrs_unix.go",
            "server_statvfs_impl.go",
            "server_statvfs_linux.go",
            "server_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "attrs_stubs.go",
            "server_stubs.go",
        ],
        "//conditions:default": [],
    }),
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
