load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "backend.go",
        "batch_tx.go",
        "config_default.go",
        "doc.go",
        "metrics.go",
        "read_tx.go",
        "tx_buffer.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "config_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "config_windows.go",
        ],
        "//conditions:default": [],
    }),
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
