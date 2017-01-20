package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["utils.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/storage/etcd/etcdtest:go_default_library",
        "//pkg/storage/etcd/testing/testingcert:go_default_library",
        "//pkg/storage/storagebackend:go_default_library",
        "//vendor:github.com/coreos/etcd/client",
        "//vendor:github.com/coreos/etcd/clientv3",
        "//vendor:github.com/coreos/etcd/etcdserver",
        "//vendor:github.com/coreos/etcd/etcdserver/api/v2http",
        "//vendor:github.com/coreos/etcd/integration",
        "//vendor:github.com/coreos/etcd/pkg/testutil",
        "//vendor:github.com/coreos/etcd/pkg/transport",
        "//vendor:github.com/coreos/etcd/pkg/types",
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/context",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
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
        "//pkg/storage/etcd/testing/testingcert:all-srcs",
    ],
    tags = ["automanaged"],
)
