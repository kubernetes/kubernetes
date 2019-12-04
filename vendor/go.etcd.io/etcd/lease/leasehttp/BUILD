load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "http.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/lease/leasehttp",
    importpath = "go.etcd.io/etcd/lease/leasehttp",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/go.etcd.io/etcd/etcdserver/etcdserverpb:go_default_library",
        "//vendor/go.etcd.io/etcd/lease:go_default_library",
        "//vendor/go.etcd.io/etcd/lease/leasepb:go_default_library",
        "//vendor/go.etcd.io/etcd/pkg/httputil:go_default_library",
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
