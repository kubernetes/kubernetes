package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["well_known_labels.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kubelet/pkg/apis",
    importpath = "k8s.io/kubelet/pkg/apis",
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
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
        "//staging/src/k8s.io/kubelet/pkg/apis/credentialprovider:all-srcs",
        "//staging/src/k8s.io/kubelet/pkg/apis/deviceplugin/v1alpha:all-srcs",
        "//staging/src/k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1:all-srcs",
        "//staging/src/k8s.io/kubelet/pkg/apis/pluginregistration/v1:all-srcs",
        "//staging/src/k8s.io/kubelet/pkg/apis/pluginregistration/v1alpha1:all-srcs",
        "//staging/src/k8s.io/kubelet/pkg/apis/pluginregistration/v1beta1:all-srcs",
        "//staging/src/k8s.io/kubelet/pkg/apis/podresources/v1:all-srcs",
        "//staging/src/k8s.io/kubelet/pkg/apis/podresources/v1alpha1:all-srcs",
        "//staging/src/k8s.io/kubelet/pkg/apis/stats/v1alpha1:all-srcs",
    ],
    tags = ["automanaged"],
)
