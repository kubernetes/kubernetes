package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "decoder.go",
        "doc.go",
        "encoder.go",
        "metrics.go",
        "multi_readcloser.go",
        "repair.go",
        "util.go",
        "wal.go",
    ],
    importpath = "k8s.io/kubernetes/third_party/forked/etcd237/wal",
    deps = [
        "//third_party/forked/etcd237/pkg/fileutil:go_default_library",
        "//vendor/github.com/coreos/etcd/pkg/crc:go_default_library",
        "//vendor/github.com/coreos/etcd/pkg/pbutil:go_default_library",
        "//vendor/github.com/coreos/etcd/raft/raftpb:go_default_library",
        "//vendor/github.com/coreos/etcd/wal/walpb:go_default_library",
        "//vendor/github.com/coreos/pkg/capnslog:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
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
        "//third_party/forked/etcd237/wal/walpb:all-srcs",
    ],
    tags = ["automanaged"],
)
