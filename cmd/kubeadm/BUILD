package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)
load("//staging/src/k8s.io/component-base/version:def.bzl", "version_x_defs")

go_binary(
    name = "kubeadm",
    embed = [":go_default_library"],
    pure = "on",
    x_defs = version_x_defs(),
)

go_library(
    name = "go_default_library",
    srcs = ["kubeadm.go"],
    importpath = "k8s.io/kubernetes/cmd/kubeadm",
    deps = [
        "//cmd/kubeadm/app:go_default_library",
        "//cmd/kubeadm/app/util:go_default_library",
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
        "//cmd/kubeadm/app:all-srcs",
        "//cmd/kubeadm/test:all-srcs",
    ],
    tags = ["automanaged"],
)
