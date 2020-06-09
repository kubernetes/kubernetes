load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["endpoint.go"],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/clientv3/balancer/resolver/endpoint",
    importpath = "go.etcd.io/etcd/clientv3/balancer/resolver/endpoint",
    visibility = ["//visibility:public"],
    deps = ["//vendor/google.golang.org/grpc/resolver:go_default_library"],
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
