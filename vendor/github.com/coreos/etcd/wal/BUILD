load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "decoder.go",
        "doc.go",
        "encoder.go",
        "file_pipeline.go",
        "metrics.go",
        "repair.go",
        "util.go",
        "wal.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "wal_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "wal_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "wal_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "wal_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "wal_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "wal_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "wal_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "wal_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "wal_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "wal_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "wal_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/coreos/etcd/wal",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/etcd/pkg/crc:go_default_library",
        "//vendor/github.com/coreos/etcd/pkg/fileutil:go_default_library",
        "//vendor/github.com/coreos/etcd/pkg/ioutil:go_default_library",
        "//vendor/github.com/coreos/etcd/pkg/pbutil:go_default_library",
        "//vendor/github.com/coreos/etcd/raft:go_default_library",
        "//vendor/github.com/coreos/etcd/raft/raftpb:go_default_library",
        "//vendor/github.com/coreos/etcd/wal/walpb:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/coreos/etcd/wal/walpb:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
