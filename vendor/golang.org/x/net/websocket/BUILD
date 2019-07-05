load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "client.go",
        "dial.go",
        "hybi.go",
        "server.go",
        "websocket.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/net/websocket",
    importpath = "golang.org/x/net/websocket",
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
