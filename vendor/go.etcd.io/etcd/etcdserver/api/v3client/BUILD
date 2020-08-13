load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "v3client.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/etcdserver/api/v3client",
    importpath = "go.etcd.io/etcd/etcdserver/api/v3client",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/go.etcd.io/etcd/clientv3:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v3rpc:go_default_library",
        "//vendor/go.etcd.io/etcd/proxy/grpcproxy/adapter:go_default_library",
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
