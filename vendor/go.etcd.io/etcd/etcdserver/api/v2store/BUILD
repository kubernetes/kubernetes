load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "event.go",
        "event_history.go",
        "event_queue.go",
        "metrics.go",
        "node.go",
        "node_extern.go",
        "stats.go",
        "store.go",
        "ttl_key_heap.go",
        "watcher.go",
        "watcher_hub.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/etcdserver/api/v2store",
    importpath = "go.etcd.io/etcd/etcdserver/api/v2store",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/jonboulle/clockwork:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v2error:go_default_library",
        "//vendor/go.etcd.io/etcd/pkg/types:go_default_library",
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
