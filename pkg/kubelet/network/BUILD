package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "network.go",
        "plugins.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/componentconfig:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/network/hostport:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//pkg/util/sysctl:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//pkg/kubelet/network/cni:all-srcs",
        "//pkg/kubelet/network/hairpin:all-srcs",
        "//pkg/kubelet/network/hostport:all-srcs",
        "//pkg/kubelet/network/kubenet:all-srcs",
        "//pkg/kubelet/network/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
