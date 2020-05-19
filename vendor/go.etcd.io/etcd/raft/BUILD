load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bootstrap.go",
        "doc.go",
        "log.go",
        "log_unstable.go",
        "logger.go",
        "node.go",
        "raft.go",
        "rawnode.go",
        "read_only.go",
        "status.go",
        "storage.go",
        "util.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/raft",
    importpath = "go.etcd.io/etcd/raft",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/go.etcd.io/etcd/raft/confchange:go_default_library",
        "//vendor/go.etcd.io/etcd/raft/quorum:go_default_library",
        "//vendor/go.etcd.io/etcd/raft/raftpb:go_default_library",
        "//vendor/go.etcd.io/etcd/raft/tracker:go_default_library",
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
        "//vendor/go.etcd.io/etcd/raft/confchange:all-srcs",
        "//vendor/go.etcd.io/etcd/raft/quorum:all-srcs",
        "//vendor/go.etcd.io/etcd/raft/raftpb:all-srcs",
        "//vendor/go.etcd.io/etcd/raft/tracker:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
