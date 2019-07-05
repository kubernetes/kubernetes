load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "index.go",
        "key_index.go",
        "kv.go",
        "kv_view.go",
        "kvstore.go",
        "kvstore_compaction.go",
        "kvstore_txn.go",
        "metrics.go",
        "metrics_txn.go",
        "revision.go",
        "util.go",
        "watchable_store.go",
        "watchable_store_txn.go",
        "watcher.go",
        "watcher_group.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/coreos/etcd/mvcc",
    importpath = "github.com/coreos/etcd/mvcc",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/etcd/lease:go_default_library",
        "//vendor/github.com/coreos/etcd/mvcc/backend:go_default_library",
        "//vendor/github.com/coreos/etcd/mvcc/mvccpb:go_default_library",
        "//vendor/github.com/coreos/etcd/pkg/adt:go_default_library",
        "//vendor/github.com/coreos/etcd/pkg/schedule:go_default_library",
        "//vendor/github.com/coreos/pkg/capnslog:go_default_library",
        "//vendor/github.com/google/btree:go_default_library",
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
        "//vendor/github.com/coreos/etcd/mvcc/backend:all-srcs",
        "//vendor/github.com/coreos/etcd/mvcc/mvccpb:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
