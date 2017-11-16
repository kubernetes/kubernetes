package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "attachlease",
    importpath = "k8s.io/kubernetes/cluster/images/etcd/attachlease",
    library = ":go_default_library",
)

go_library(
    name = "go_default_library",
    srcs = ["attachlease.go"],
    importpath = "k8s.io/kubernetes/cluster/images/etcd/attachlease",
    deps = [
        "//vendor/github.com/coreos/etcd/clientv3:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
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
)
