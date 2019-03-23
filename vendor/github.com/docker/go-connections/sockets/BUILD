load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "inmem_socket.go",
        "proxy.go",
        "sockets.go",
        "sockets_unix.go",
        "sockets_windows.go",
        "tcp_socket.go",
        "unix_socket.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/go-connections/sockets",
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
