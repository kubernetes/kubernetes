load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "cluster.go",
        "doc.go",
        "errors.go",
        "member.go",
        "metrics.go",
        "store.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/etcdserver/api/membership",
    importpath = "go.etcd.io/etcd/etcdserver/api/membership",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/go-semver/semver:go_default_library",
        "//vendor/github.com/coreos/pkg/capnslog:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v2error:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v2store:go_default_library",
        "//vendor/go.etcd.io/etcd/mvcc/backend:go_default_library",
        "//vendor/go.etcd.io/etcd/pkg/netutil:go_default_library",
        "//vendor/go.etcd.io/etcd/pkg/types:go_default_library",
        "//vendor/go.etcd.io/etcd/raft:go_default_library",
        "//vendor/go.etcd.io/etcd/raft/raftpb:go_default_library",
        "//vendor/go.etcd.io/etcd/version:go_default_library",
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
