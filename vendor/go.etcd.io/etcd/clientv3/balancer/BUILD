load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "balancer.go",
        "utils.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/clientv3/balancer",
    importpath = "go.etcd.io/etcd/clientv3/balancer",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/go.etcd.io/etcd/clientv3/balancer/connectivity:go_default_library",
        "//vendor/go.etcd.io/etcd/clientv3/balancer/picker:go_default_library",
        "//vendor/go.uber.org/zap:go_default_library",
        "//vendor/google.golang.org/grpc/balancer:go_default_library",
        "//vendor/google.golang.org/grpc/connectivity:go_default_library",
        "//vendor/google.golang.org/grpc/resolver:go_default_library",
        "//vendor/google.golang.org/grpc/resolver/dns:go_default_library",
        "//vendor/google.golang.org/grpc/resolver/passthrough:go_default_library",
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
        "//vendor/go.etcd.io/etcd/clientv3/balancer/connectivity:all-srcs",
        "//vendor/go.etcd.io/etcd/clientv3/balancer/picker:all-srcs",
        "//vendor/go.etcd.io/etcd/clientv3/balancer/resolver/endpoint:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
