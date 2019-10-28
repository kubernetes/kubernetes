load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "auth_client_adapter.go",
        "chan_stream.go",
        "cluster_client_adapter.go",
        "doc.go",
        "election_client_adapter.go",
        "kv_client_adapter.go",
        "lease_client_adapter.go",
        "lock_client_adapter.go",
        "maintenance_client_adapter.go",
        "watch_client_adapter.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/proxy/grpcproxy/adapter",
    importpath = "go.etcd.io/etcd/proxy/grpcproxy/adapter",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/go.etcd.io/etcd/etcdserver/api/v3election/v3electionpb:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v3lock/v3lockpb:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/etcdserverpb:go_default_library",
        "//vendor/google.golang.org/grpc:go_default_library",
        "//vendor/google.golang.org/grpc/codes:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
