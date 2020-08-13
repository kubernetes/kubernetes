load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["alarms.go"],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/etcdserver/api/v3alarm",
    importpath = "go.etcd.io/etcd/etcdserver/api/v3alarm",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/pkg/capnslog:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/etcdserverpb:go_default_library",
        "//vendor/go.etcd.io/etcd/mvcc/backend:go_default_library",
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
