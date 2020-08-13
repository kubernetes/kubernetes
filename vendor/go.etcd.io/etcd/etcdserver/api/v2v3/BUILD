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
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/etcdserver/api/v2v3",
    importpath = "go.etcd.io/etcd/etcdserver/api/v2v3",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/go-semver/semver:go_default_library",
        "//vendor/go.etcd.io/etcd/clientv3:go_default_library",
        "//vendor/go.etcd.io/etcd/clientv3/concurrency:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/membership:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v2error:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v2store:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/etcdserverpb:go_default_library",
        "//vendor/go.etcd.io/etcd/mvcc/mvccpb:go_default_library",
        "//vendor/go.etcd.io/etcd/pkg/types:go_default_library",
        "//vendor/go.uber.org/zap:go_default_library",
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
