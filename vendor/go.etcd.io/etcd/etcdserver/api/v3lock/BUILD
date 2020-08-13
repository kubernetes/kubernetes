load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "lock.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/etcdserver/api/v3lock",
    importpath = "go.etcd.io/etcd/etcdserver/api/v3lock",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/go.etcd.io/etcd/clientv3:go_default_library",
        "//vendor/go.etcd.io/etcd/clientv3/concurrency:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v3lock/v3lockpb:go_default_library",
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
        "//vendor/go.etcd.io/etcd/etcdserver/api/v3lock/v3lockpb:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
