load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "election.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/etcdserver/api/v3election",
    importpath = "go.etcd.io/etcd/etcdserver/api/v3election",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/go.etcd.io/etcd/clientv3:go_default_library",
        "//vendor/go.etcd.io/etcd/clientv3/concurrency:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v3election/v3electionpb:go_default_library",
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
        "//vendor/go.etcd.io/etcd/etcdserver/api/v3election/v3electionpb:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
