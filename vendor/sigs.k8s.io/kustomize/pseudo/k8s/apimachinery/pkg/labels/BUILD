load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "labels.go",
        "selector.go",
        "zz_generated.deepcopy.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/labels",
    importpath = "sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/labels",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/k8s.io/klog:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/selection:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/validation:go_default_library",
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
