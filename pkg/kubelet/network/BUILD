package(default_visibility = ["//visibility:public"])

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
    deps = [
        "//pkg/kubelet/apis/kubeletconfig:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/network/hostport:go_default_library",
        "//pkg/util/sysctl:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/validation:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/k8s.io/utils/exec:go_default_library",
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
