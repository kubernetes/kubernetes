load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["etcd.go"],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/util/etcd",
    visibility = ["//visibility:public"],
    deps = [
        "//cmd/kubeadm/app/constants:go_default_library",
        "//cmd/kubeadm/app/util/staticpod:go_default_library",
        "//vendor/github.com/coreos/etcd/clientv3:go_default_library",
        "//vendor/github.com/coreos/etcd/pkg/transport:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["etcd_test.go"],
    embed = [":go_default_library"],
    deps = ["//cmd/kubeadm/test:go_default_library"],
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
