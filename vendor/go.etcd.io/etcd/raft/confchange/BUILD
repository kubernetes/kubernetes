load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "confchange.go",
        "restore.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/raft/confchange",
    importpath = "go.etcd.io/etcd/raft/confchange",
    visibility = ["//visibility:public"],
    deps = [
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
