load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "coder.go",
        "doc.go",
        "http.go",
        "metrics.go",
        "msg_codec.go",
        "msgappv2_codec.go",
        "peer.go",
        "peer_status.go",
        "pipeline.go",
        "probing_status.go",
        "remote.go",
        "snapshot_sender.go",
        "stream.go",
        "transport.go",
        "urlpick.go",
        "util.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/etcdserver/api/rafthttp",
    importpath = "go.etcd.io/etcd/etcdserver/api/rafthttp",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/go-semver/semver:go_default_library",
        "//vendor/github.com/coreos/pkg/capnslog:go_default_library",
        "//vendor/github.com/dustin/go-humanize:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/github.com/xiang90/probing:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/snap:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v2stats:go_default_library",
        "//vendor/go.etcd.io/etcd/pkg/httputil:go_default_library",
        "//vendor/go.etcd.io/etcd/pkg/ioutil:go_default_library",
        "//vendor/go.etcd.io/etcd/pkg/logutil:go_default_library",
        "//vendor/go.etcd.io/etcd/pkg/pbutil:go_default_library",
        "//vendor/go.etcd.io/etcd/pkg/transport:go_default_library",
        "//vendor/go.etcd.io/etcd/pkg/types:go_default_library",
        "//vendor/go.etcd.io/etcd/raft:go_default_library",
        "//vendor/go.etcd.io/etcd/raft/raftpb:go_default_library",
        "//vendor/go.etcd.io/etcd/version:go_default_library",
        "//vendor/go.uber.org/zap:go_default_library",
        "//vendor/golang.org/x/time/rate:go_default_library",
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
