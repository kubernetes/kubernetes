package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["cidr_set_test.go"],
    embed = [":go_default_library"],
    deps = ["//vendor/k8s.io/klog:go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = ["cidr_set.go"],
    importpath = "k8s.io/kubernetes/pkg/controller/nodeipam/ipam/cidrset",
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
