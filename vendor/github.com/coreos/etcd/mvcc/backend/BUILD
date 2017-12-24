load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "backend.go",
        "batch_tx.go",
        "doc.go",
        "metrics.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "boltoption_default.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "boltoption_default.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "boltoption_default.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "boltoption_default.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "boltoption_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "boltoption_default.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "boltoption_default.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "boltoption_default.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "boltoption_default.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "boltoption_default.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "boltoption_default.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/coreos/etcd/mvcc/backend",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/boltdb/bolt:go_default_library",
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
