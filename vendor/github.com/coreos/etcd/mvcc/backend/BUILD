load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "backend.go",
        "batch_tx.go",
        "doc.go",
        "metrics.go",
        "read_tx.go",
        "tx_buffer.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "config_default.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "config_default.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "config_default.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "config_default.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "config_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "config_default.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "config_default.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "config_default.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "config_default.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "config_default.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
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
