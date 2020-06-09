load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["cpuset.go"],
    importpath = "k8s.io/kubernetes/pkg/kubelet/cm/cpuset",
    visibility = ["//visibility:public"],
    deps = ["//vendor/k8s.io/klog/v2:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = ["cpuset_test.go"],
    embed = [":go_default_library"],
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
