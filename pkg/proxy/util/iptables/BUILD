load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["traffic.go"],
    importpath = "k8s.io/kubernetes/pkg/proxy/util/iptables",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/util/iptables:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
        "//vendor/k8s.io/utils/net:go_default_library",
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
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["traffic_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/util/iptables:go_default_library",
        "//pkg/util/iptables/testing:go_default_library",
    ],
)
