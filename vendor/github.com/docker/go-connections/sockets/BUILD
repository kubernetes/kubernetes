load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "inmem_socket.go",
        "proxy.go",
        "sockets.go",
        "sockets_unix.go",
        "tcp_socket.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "unix_socket.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "sockets_windows.go",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/net/proxy:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "//vendor/github.com/Sirupsen/logrus:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
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
