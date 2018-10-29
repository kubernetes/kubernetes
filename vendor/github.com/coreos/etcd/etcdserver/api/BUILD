load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "capability.go",
        "cluster.go",
        "doc.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/coreos/etcd/etcdserver/api",
    importpath = "github.com/coreos/etcd/etcdserver/api",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/etcd/etcdserver/membership:go_default_library",
        "//vendor/github.com/coreos/etcd/pkg/types:go_default_library",
        "//vendor/github.com/coreos/etcd/version:go_default_library",
        "//vendor/github.com/coreos/go-semver/semver:go_default_library",
        "//vendor/github.com/coreos/pkg/capnslog:go_default_library",
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
        "//vendor/github.com/coreos/etcd/etcdserver/api/etcdhttp:all-srcs",
        "//vendor/github.com/coreos/etcd/etcdserver/api/v2http:all-srcs",
        "//vendor/github.com/coreos/etcd/etcdserver/api/v2v3:all-srcs",
        "//vendor/github.com/coreos/etcd/etcdserver/api/v3client:all-srcs",
        "//vendor/github.com/coreos/etcd/etcdserver/api/v3election:all-srcs",
        "//vendor/github.com/coreos/etcd/etcdserver/api/v3lock:all-srcs",
        "//vendor/github.com/coreos/etcd/etcdserver/api/v3rpc:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
