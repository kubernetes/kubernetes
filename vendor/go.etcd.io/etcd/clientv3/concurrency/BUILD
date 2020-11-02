load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "election.go",
        "key.go",
        "mutex.go",
        "session.go",
        "stm.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/clientv3/concurrency",
    importpath = "go.etcd.io/etcd/clientv3/concurrency",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/go.etcd.io/etcd/clientv3:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/etcdserverpb:go_default_library",
        "//vendor/go.etcd.io/etcd/mvcc/mvccpb:go_default_library",
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
