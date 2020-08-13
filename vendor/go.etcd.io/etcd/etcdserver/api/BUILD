load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "capability.go",
        "cluster.go",
        "doc.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/etcdserver/api",
    importpath = "go.etcd.io/etcd/etcdserver/api",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/go-semver/semver:go_default_library",
        "//vendor/github.com/coreos/pkg/capnslog:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/membership:go_default_library",
        "//vendor/go.etcd.io/etcd/pkg/types:go_default_library",
        "//vendor/go.etcd.io/etcd/version:go_default_library",
        "//vendor/go.uber.org/zap:go_default_library",
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
        "//vendor/go.etcd.io/etcd/etcdserver/api/etcdhttp:all-srcs",
        "//vendor/go.etcd.io/etcd/etcdserver/api/membership:all-srcs",
        "//vendor/go.etcd.io/etcd/etcdserver/api/rafthttp:all-srcs",
        "//vendor/go.etcd.io/etcd/etcdserver/api/snap:all-srcs",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v2auth:all-srcs",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v2discovery:all-srcs",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v2error:all-srcs",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v2http:all-srcs",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v2stats:all-srcs",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v2store:all-srcs",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v2v3:all-srcs",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v3alarm:all-srcs",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v3client:all-srcs",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v3compactor:all-srcs",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v3election:all-srcs",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v3lock:all-srcs",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v3rpc:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
