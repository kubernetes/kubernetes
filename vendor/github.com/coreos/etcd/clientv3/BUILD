load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "auth.go",
        "client.go",
        "cluster.go",
        "compact_op.go",
        "compare.go",
        "config.go",
        "doc.go",
        "health_balancer.go",
        "kv.go",
        "lease.go",
        "logger.go",
        "maintenance.go",
        "op.go",
        "options.go",
        "ready_wait.go",
        "retry.go",
        "sort.go",
        "txn.go",
        "watch.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/coreos/etcd/clientv3",
    importpath = "github.com/coreos/etcd/clientv3",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/etcd/auth/authpb:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver/etcdserverpb:go_default_library",
        "//vendor/github.com/coreos/etcd/mvcc/mvccpb:go_default_library",
        "//vendor/github.com/coreos/etcd/pkg/types:go_default_library",
        "//vendor/google.golang.org/grpc:go_default_library",
        "//vendor/google.golang.org/grpc/codes:go_default_library",
        "//vendor/google.golang.org/grpc/credentials:go_default_library",
        "//vendor/google.golang.org/grpc/grpclog:go_default_library",
        "//vendor/google.golang.org/grpc/health/grpc_health_v1:go_default_library",
        "//vendor/google.golang.org/grpc/keepalive:go_default_library",
        "//vendor/google.golang.org/grpc/metadata:go_default_library",
        "//vendor/google.golang.org/grpc/status:go_default_library",
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
        "//vendor/github.com/coreos/etcd/clientv3/concurrency:all-srcs",
        "//vendor/github.com/coreos/etcd/clientv3/namespace:all-srcs",
        "//vendor/github.com/coreos/etcd/clientv3/naming:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
