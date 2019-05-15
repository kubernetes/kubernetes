load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "network.go",
        "plugins.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/kubelet/pkg/dockershim/network",
    visibility = ["//visibility:public"],
    deps = [
        "//cmd/kubelet/pkg/apis/config:go_default_library",
        "//cmd/kubelet/pkg/container:go_default_library",
        "//cmd/kubelet/pkg/dockershim/network/hostport:go_default_library",
        "//cmd/kubelet/pkg/dockershim/network/metrics:go_default_library",
        "//pkg/util/sysctl:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
        "//cmd/kubelet/pkg/dockershim/network/cni:all-srcs",
        "//cmd/kubelet/pkg/dockershim/network/hairpin:all-srcs",
        "//cmd/kubelet/pkg/dockershim/network/hostport:all-srcs",
        "//cmd/kubelet/pkg/dockershim/network/kubenet:all-srcs",
        "//cmd/kubelet/pkg/dockershim/network/metrics:all-srcs",
        "//cmd/kubelet/pkg/dockershim/network/testing:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
