load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "client.go",
        "client_clone.go",
        "client_clone_legacy.go",
        "compression.go",
        "conn.go",
        "conn_write.go",
        "conn_write_legacy.go",
        "doc.go",
        "json.go",
        "mask.go",
        "prepared.go",
        "proxy.go",
        "server.go",
        "trace.go",
        "trace_17.go",
        "util.go",
        "x_net_proxy.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/gorilla/websocket",
    importpath = "github.com/gorilla/websocket",
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
