load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bridge.go",
        "cluster.go",
        "cluster_direct.go",
        "doc.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/integration",
    importpath = "go.etcd.io/etcd/integration",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/soheilhy/cmux:go_default_library",
        "//vendor/go.etcd.io/etcd/client:go_default_library",
        "//vendor/go.etcd.io/etcd/clientv3:go_default_library",
        "//vendor/go.etcd.io/etcd/embed:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/etcdhttp:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/rafthttp:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v2http:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v3client:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v3election:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v3election/v3electionpb:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v3lock:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v3lock/v3lockpb:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v3rpc:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/etcdserverpb:go_default_library",
        "//vendor/go.etcd.io/etcd/pkg/logutil:go_default_library",
        "//vendor/go.etcd.io/etcd/pkg/testutil:go_default_library",
        "//vendor/go.etcd.io/etcd/pkg/tlsutil:go_default_library",
        "//vendor/go.etcd.io/etcd/pkg/transport:go_default_library",
        "//vendor/go.etcd.io/etcd/pkg/types:go_default_library",
        "//vendor/go.uber.org/zap:go_default_library",
        "//vendor/golang.org/x/crypto/bcrypt:go_default_library",
        "//vendor/google.golang.org/grpc:go_default_library",
        "//vendor/google.golang.org/grpc/grpclog:go_default_library",
        "//vendor/google.golang.org/grpc/keepalive:go_default_library",
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
