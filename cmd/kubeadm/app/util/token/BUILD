package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["tokens_test.go"],
    embed = [":go_default_library"],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/util/token",
    deps = ["//cmd/kubeadm/app/apis/kubeadm:go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = ["tokens.go"],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/util/token",
    deps = ["//cmd/kubeadm/app/apis/kubeadm:go_default_library"],
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
