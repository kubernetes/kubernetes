package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["runtime_test.go"],
    embed = [":go_default_library"],
    deps = ["//vendor/k8s.io/klog:go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = ["runtime.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apimachinery/pkg/util/runtime",
    importpath = "k8s.io/apimachinery/pkg/util/runtime",
    deps = ["//vendor/k8s.io/klog:go_default_library"],
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
