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
    tags = ["automanaged"],
    deps = [
        "//third_party/forked/etcd237/pkg/fileutil:go_default_library",
        "//vendor:github.com/coreos/etcd/pkg/crc",
        "//vendor:github.com/coreos/etcd/pkg/pbutil",
        "//vendor:github.com/coreos/etcd/raft/raftpb",
        "//vendor:github.com/coreos/etcd/wal/walpb",
        "//vendor:github.com/coreos/pkg/capnslog",
        "//vendor:github.com/prometheus/client_golang/prometheus",
    ],
)
