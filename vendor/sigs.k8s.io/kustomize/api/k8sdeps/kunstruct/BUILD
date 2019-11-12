load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "factory.go",
        "hasher.go",
        "helper.go",
        "unstructadapter.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/api/k8sdeps/kunstruct",
    importpath = "sigs.k8s.io/kustomize/api/k8sdeps/kunstruct",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/evanphx/json-patch:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/hasher:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/ifc:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/internal/k8sdeps/configmapandsecret:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/resid:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/types:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/api/core/v1:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/apis/meta/v1/unstructured:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/labels:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/runtime:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/strategicpatch:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/yaml:go_default_library",
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
