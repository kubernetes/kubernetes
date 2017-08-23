load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "log.go",
        "log_unstable.go",
        "logger.go",
        "node.go",
        "progress.go",
        "raft.go",
        "rawnode.go",
        "read_only.go",
        "status.go",
        "storage.go",
        "util.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/etcd/raft/raftpb:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
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
        "//vendor/github.com/coreos/etcd/raft/raftpb:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
