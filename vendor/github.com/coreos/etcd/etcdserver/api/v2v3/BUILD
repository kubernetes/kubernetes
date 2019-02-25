load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "cluster.go",
        "doc.go",
        "server.go",
        "store.go",
        "watcher.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/coreos/etcd/etcdserver/api/v2v3",
    importpath = "github.com/coreos/etcd/etcdserver/api/v2v3",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/etcd/clientv3:go_default_library",
        "//vendor/github.com/coreos/etcd/clientv3/concurrency:go_default_library",
        "//vendor/github.com/coreos/etcd/error:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver/api:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver/etcdserverpb:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver/membership:go_default_library",
        "//vendor/github.com/coreos/etcd/mvcc/mvccpb:go_default_library",
        "//vendor/github.com/coreos/etcd/pkg/types:go_default_library",
        "//vendor/github.com/coreos/etcd/store:go_default_library",
        "//vendor/github.com/coreos/go-semver/semver:go_default_library",
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
