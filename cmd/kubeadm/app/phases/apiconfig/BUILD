package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["clusterroles.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/kubeadm/app/master:go_default_library",
        "//pkg/apis/rbac/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
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
