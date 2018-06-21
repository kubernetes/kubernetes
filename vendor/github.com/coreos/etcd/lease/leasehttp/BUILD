load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "http.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/coreos/etcd/lease/leasehttp",
    importpath = "github.com/coreos/etcd/lease/leasehttp",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/etcd/etcdserver/etcdserverpb:go_default_library",
        "//vendor/github.com/coreos/etcd/lease:go_default_library",
        "//vendor/github.com/coreos/etcd/lease/leasepb:go_default_library",
        "//vendor/github.com/coreos/etcd/pkg/httputil:go_default_library",
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
