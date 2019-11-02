load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["credentials.go"],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/clientv3/credentials",
    importpath = "go.etcd.io/etcd/clientv3/credentials",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/go.etcd.io/etcd/clientv3/balancer/resolver/endpoint:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v3rpc/rpctypes:go_default_library",
        "//vendor/google.golang.org/grpc/credentials:go_default_library",
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
