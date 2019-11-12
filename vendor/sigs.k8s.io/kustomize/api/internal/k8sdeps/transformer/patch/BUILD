load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["conflictdetector.go"],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/api/internal/k8sdeps/transformer/patch",
    importpath = "sigs.k8s.io/kustomize/api/internal/k8sdeps/transformer/patch",
    visibility = ["//vendor/sigs.k8s.io/kustomize/api:__subpackages__"],
    deps = [
        "//vendor/github.com/evanphx/json-patch:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/resid:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/resmap:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/resource:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/runtime:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/mergepatch:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/strategicpatch:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/client-go/kubernetes/scheme:go_default_library",
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
