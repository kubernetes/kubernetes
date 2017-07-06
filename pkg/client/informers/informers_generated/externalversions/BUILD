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
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions/admissionregistration:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions/apps:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions/autoscaling:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions/batch:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions/certificates:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions/core:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions/extensions:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions/internalinterfaces:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions/networking:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions/policy:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions/rbac:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions/settings:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions/storage:go_default_library",
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
        "//vendor/k8s.io/client-go/tools/cache:go_default_library",
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
        "//pkg/client/informers/informers_generated/externalversions/admissionregistration:all-srcs",
        "//pkg/client/informers/informers_generated/externalversions/apps:all-srcs",
        "//pkg/client/informers/informers_generated/externalversions/autoscaling:all-srcs",
        "//pkg/client/informers/informers_generated/externalversions/batch:all-srcs",
        "//pkg/client/informers/informers_generated/externalversions/certificates:all-srcs",
        "//pkg/client/informers/informers_generated/externalversions/core:all-srcs",
        "//pkg/client/informers/informers_generated/externalversions/extensions:all-srcs",
        "//pkg/client/informers/informers_generated/externalversions/internalinterfaces:all-srcs",
        "//pkg/client/informers/informers_generated/externalversions/networking:all-srcs",
        "//pkg/client/informers/informers_generated/externalversions/policy:all-srcs",
        "//pkg/client/informers/informers_generated/externalversions/rbac:all-srcs",
        "//pkg/client/informers/informers_generated/externalversions/settings:all-srcs",
        "//pkg/client/informers/informers_generated/externalversions/storage:all-srcs",
    ],
    tags = ["automanaged"],
)
