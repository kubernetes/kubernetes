load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "backend.go",
        "batch_tx.go",
        "config_default.go",
        "config_linux.go",
        "config_windows.go",
        "doc.go",
        "metrics.go",
        "read_tx.go",
        "tx_buffer.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/coreos/etcd/mvcc/backend",
    importpath = "github.com/coreos/etcd/mvcc/backend",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/bbolt:go_default_library",
        "//vendor/github.com/coreos/pkg/capnslog:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
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
