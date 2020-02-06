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
    importpath = "k8s.io/kubernetes/pkg/kubelet/apis",
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
        "//pkg/kubelet/apis/config:all-srcs",
        "//pkg/kubelet/apis/podresources:all-srcs",
        "//pkg/kubelet/apis/resourcemetrics/v1alpha1:all-srcs",
        "//pkg/kubelet/apis/stats/v1alpha1:all-srcs",
    ],
    tags = ["automanaged"],
)
