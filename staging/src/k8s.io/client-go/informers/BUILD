package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "factory.go",
        "generic.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/k8s.io/api/admissionregistration/v1alpha1:go_default_library",
        "//vendor/k8s.io/api/apps/v1beta1:go_default_library",
        "//vendor/k8s.io/api/autoscaling/v1:go_default_library",
        "//vendor/k8s.io/api/autoscaling/v2alpha1:go_default_library",
        "//vendor/k8s.io/api/batch/v1:go_default_library",
        "//vendor/k8s.io/api/batch/v2alpha1:go_default_library",
        "//vendor/k8s.io/api/certificates/v1beta1:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/api/extensions/v1beta1:go_default_library",
        "//vendor/k8s.io/api/networking/v1:go_default_library",
        "//vendor/k8s.io/api/policy/v1beta1:go_default_library",
        "//vendor/k8s.io/api/rbac/v1alpha1:go_default_library",
        "//vendor/k8s.io/api/rbac/v1beta1:go_default_library",
        "//vendor/k8s.io/api/settings/v1alpha1:go_default_library",
        "//vendor/k8s.io/api/storage/v1:go_default_library",
        "//vendor/k8s.io/api/storage/v1beta1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/client-go/informers/admissionregistration:go_default_library",
        "//vendor/k8s.io/client-go/informers/apps:go_default_library",
        "//vendor/k8s.io/client-go/informers/autoscaling:go_default_library",
        "//vendor/k8s.io/client-go/informers/batch:go_default_library",
        "//vendor/k8s.io/client-go/informers/certificates:go_default_library",
        "//vendor/k8s.io/client-go/informers/core:go_default_library",
        "//vendor/k8s.io/client-go/informers/extensions:go_default_library",
        "//vendor/k8s.io/client-go/informers/internalinterfaces:go_default_library",
        "//vendor/k8s.io/client-go/informers/networking:go_default_library",
        "//vendor/k8s.io/client-go/informers/policy:go_default_library",
        "//vendor/k8s.io/client-go/informers/rbac:go_default_library",
        "//vendor/k8s.io/client-go/informers/settings:go_default_library",
        "//vendor/k8s.io/client-go/informers/storage:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/k8s.io/client-go/tools/cache:go_default_library",
    ],
)
