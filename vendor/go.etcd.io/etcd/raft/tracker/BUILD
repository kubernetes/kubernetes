load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "inflights.go",
        "progress.go",
        "state.go",
        "tracker.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/raft/tracker",
    importpath = "go.etcd.io/etcd/raft/tracker",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/go.etcd.io/etcd/raft/quorum:go_default_library",
        "//vendor/go.etcd.io/etcd/raft/raftpb:go_default_library",
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
