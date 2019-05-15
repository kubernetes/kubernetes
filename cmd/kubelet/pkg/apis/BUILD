package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "well_known_annotations.go",
        "well_known_annotations_windows.go",
        "well_known_labels.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/kubelet/pkg/apis",
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//pkg/features:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        ],
        "//conditions:default": [],
    }),
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
        "//cmd/kubelet/pkg/apis/config:all-srcs",
        "//cmd/kubelet/pkg/apis/deviceplugin/v1alpha:all-srcs",
        "//cmd/kubelet/pkg/apis/deviceplugin/v1beta1:all-srcs",
        "//cmd/kubelet/pkg/apis/pluginregistration/v1:all-srcs",
        "//cmd/kubelet/pkg/apis/pluginregistration/v1alpha1:all-srcs",
        "//cmd/kubelet/pkg/apis/pluginregistration/v1beta1:all-srcs",
        "//cmd/kubelet/pkg/apis/podresources:all-srcs",
        "//cmd/kubelet/pkg/apis/resourcemetrics/v1alpha1:all-srcs",
        "//cmd/kubelet/pkg/apis/stats/v1alpha1:all-srcs",
    ],
    tags = ["automanaged"],
)
