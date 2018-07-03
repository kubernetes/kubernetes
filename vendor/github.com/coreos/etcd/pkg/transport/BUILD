load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "keepalive_listener.go",
        "limit_listen.go",
        "listener.go",
        "listener_tls.go",
        "timeout_conn.go",
        "timeout_dialer.go",
        "timeout_listener.go",
        "timeout_transport.go",
        "tls.go",
        "transport.go",
        "unix_listener.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/coreos/etcd/pkg/transport",
    importpath = "github.com/coreos/etcd/pkg/transport",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/coreos/etcd/pkg/tlsutil:go_default_library"],
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
