load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "db.go",
        "doc.go",
        "message.go",
        "metrics.go",
        "snapshotter.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/etcdserver/api/snap",
    importpath = "go.etcd.io/etcd/etcdserver/api/snap",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/pkg/capnslog:go_default_library",
        "//vendor/github.com/dustin/go-humanize:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/snap/snappb:go_default_library",
        "//vendor/go.etcd.io/etcd/pkg/fileutil:go_default_library",
        "//vendor/go.etcd.io/etcd/pkg/ioutil:go_default_library",
        "//vendor/go.etcd.io/etcd/pkg/pbutil:go_default_library",
        "//vendor/go.etcd.io/etcd/raft:go_default_library",
        "//vendor/go.etcd.io/etcd/raft/raftpb:go_default_library",
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
        "//vendor/go.etcd.io/etcd/etcdserver/api/snap/snappb:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
