package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["kubeadm.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm/install:go_default_library",
        "//cmd/kubeadm/app/cmd:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//vendor:github.com/spf13/pflag",
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
        "//cmd/kubeadm/app/apis/kubeadm:all-srcs",
        "//cmd/kubeadm/app/cmd:all-srcs",
        "//cmd/kubeadm/app/discovery:all-srcs",
        "//cmd/kubeadm/app/images:all-srcs",
        "//cmd/kubeadm/app/master:all-srcs",
        "//cmd/kubeadm/app/node:all-srcs",
        "//cmd/kubeadm/app/phases/certs:all-srcs",
        "//cmd/kubeadm/app/phases/kubeconfig:all-srcs",
        "//cmd/kubeadm/app/preflight:all-srcs",
        "//cmd/kubeadm/app/util:all-srcs",
    ],
    tags = ["automanaged"],
)
