package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["constants.go"],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/constants",
    deps = [
        "//pkg/bootstrap/api:go_default_library",
        "//pkg/registry/core/service/ipallocator:go_default_library",
        "//pkg/util/version:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
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

go_test(
    name = "go_default_test",
    srcs = ["constants_test.go"],
    embed = [":go_default_library"],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/constants",
    deps = ["//pkg/util/version:go_default_library"],
)
