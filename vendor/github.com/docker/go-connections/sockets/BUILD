load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "inmem_socket.go",
        "proxy.go",
        "sockets.go",
        "tcp_socket.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "sockets_unix.go",
            "unix_socket.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "sockets_unix.go",
            "unix_socket.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "sockets_unix.go",
            "unix_socket.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "sockets_unix.go",
            "unix_socket.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "sockets_unix.go",
            "unix_socket.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "sockets_unix.go",
            "unix_socket.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "sockets_unix.go",
            "unix_socket.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "sockets_unix.go",
            "unix_socket.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "sockets_unix.go",
            "unix_socket.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "sockets_unix.go",
            "unix_socket.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "sockets_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/docker/go-connections/sockets",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/net/proxy:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/Microsoft/go-winio:go_default_library",
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
