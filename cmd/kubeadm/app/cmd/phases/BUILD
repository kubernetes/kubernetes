load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["util.go"],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases",
    visibility = ["//visibility:public"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm/v1beta2:go_default_library",
        "//cmd/kubeadm/app/version:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["util_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm/v1beta2:go_default_library",
        "//cmd/kubeadm/app/version:go_default_library",
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
        "//cmd/kubeadm/app/cmd/phases/init:all-srcs",
        "//cmd/kubeadm/app/cmd/phases/join:all-srcs",
        "//cmd/kubeadm/app/cmd/phases/reset:all-srcs",
        "//cmd/kubeadm/app/cmd/phases/upgrade/node:all-srcs",
        "//cmd/kubeadm/app/cmd/phases/workflow:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
