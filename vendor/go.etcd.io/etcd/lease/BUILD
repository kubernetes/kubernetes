load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "lease_queue.go",
        "lessor.go",
        "metrics.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/lease",
    importpath = "go.etcd.io/etcd/lease",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/etcdserverpb:go_default_library",
        "//vendor/go.etcd.io/etcd/lease/leasepb:go_default_library",
        "//vendor/go.etcd.io/etcd/mvcc/backend:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/go.etcd.io/etcd/lease/leasehttp:all-srcs",
        "//vendor/go.etcd.io/etcd/lease/leasepb:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
