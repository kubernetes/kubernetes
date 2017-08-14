package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "rollback",
    library = ":go_default_library",
)

go_library(
    name = "go_default_library",
    srcs = ["rollback.go"],
    deps = [
        "//third_party/forked/etcd221/wal:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver/etcdserverpb:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver/membership:go_default_library",
        "//vendor/github.com/coreos/etcd/mvcc/backend:go_default_library",
        "//vendor/github.com/coreos/etcd/mvcc/mvccpb:go_default_library",
        "//vendor/github.com/coreos/etcd/pkg/pbutil:go_default_library",
        "//vendor/github.com/coreos/etcd/pkg/types:go_default_library",
        "//vendor/github.com/coreos/etcd/raft/raftpb:go_default_library",
        "//vendor/github.com/coreos/etcd/snap:go_default_library",
        "//vendor/github.com/coreos/etcd/store:go_default_library",
        "//vendor/github.com/coreos/etcd/wal:go_default_library",
        "//vendor/github.com/coreos/etcd/wal/walpb:go_default_library",
        "//vendor/github.com/coreos/go-semver/semver:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
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
)
