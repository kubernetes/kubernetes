load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["store.go"],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/proxy/grpcproxy/cache",
    importpath = "go.etcd.io/etcd/proxy/grpcproxy/cache",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/golang/groupcache/lru:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v3rpc/rpctypes:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/etcdserverpb:go_default_library",
        "//vendor/go.etcd.io/etcd/pkg/adt:go_default_library",
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
