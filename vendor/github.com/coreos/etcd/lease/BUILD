load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "lessor.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/coreos/etcd/lease",
    importpath = "github.com/coreos/etcd/lease",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/etcd/lease/leasepb:go_default_library",
        "//vendor/github.com/coreos/etcd/mvcc/backend:go_default_library",
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
        "//vendor/github.com/coreos/etcd/lease/leasehttp:all-srcs",
        "//vendor/github.com/coreos/etcd/lease/leasepb:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
