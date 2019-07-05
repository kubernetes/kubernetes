load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "leader.go",
        "queue.go",
        "server.go",
        "stats.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/coreos/etcd/etcdserver/stats",
    importpath = "github.com/coreos/etcd/etcdserver/stats",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/etcd/raft:go_default_library",
        "//vendor/github.com/coreos/pkg/capnslog:go_default_library",
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
