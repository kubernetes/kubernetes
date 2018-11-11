load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "auth.go",
        "cluster.go",
        "doc.go",
        "election.go",
        "health.go",
        "kv.go",
        "leader.go",
        "lease.go",
        "lock.go",
        "logger.go",
        "maintenance.go",
        "metrics.go",
        "register.go",
        "util.go",
        "watch.go",
        "watch_broadcast.go",
        "watch_broadcasts.go",
        "watch_ranges.go",
        "watcher.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/coreos/etcd/proxy/grpcproxy",
    importpath = "github.com/coreos/etcd/proxy/grpcproxy",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/etcd/clientv3:go_default_library",
        "//vendor/github.com/coreos/etcd/clientv3/concurrency:go_default_library",
        "//vendor/github.com/coreos/etcd/clientv3/naming:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver/api/etcdhttp:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver/api/v3election/v3electionpb:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver/api/v3lock/v3lockpb:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver/api/v3rpc:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver/etcdserverpb:go_default_library",
        "//vendor/github.com/coreos/etcd/mvcc:go_default_library",
        "//vendor/github.com/coreos/etcd/mvcc/mvccpb:go_default_library",
        "//vendor/github.com/coreos/etcd/proxy/grpcproxy/cache:go_default_library",
        "//vendor/github.com/coreos/pkg/capnslog:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/golang.org/x/time/rate:go_default_library",
        "//vendor/google.golang.org/grpc:go_default_library",
        "//vendor/google.golang.org/grpc/metadata:go_default_library",
        "//vendor/google.golang.org/grpc/naming:go_default_library",
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
        "//vendor/github.com/coreos/etcd/proxy/grpcproxy/adapter:all-srcs",
        "//vendor/github.com/coreos/etcd/proxy/grpcproxy/cache:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
