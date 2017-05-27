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
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/admissionregistration/v1alpha1:go_default_library",
        "//pkg/apis/apps/v1beta1:go_default_library",
        "//pkg/apis/autoscaling/v1:go_default_library",
        "//pkg/apis/autoscaling/v2alpha1:go_default_library",
        "//pkg/apis/batch/v1:go_default_library",
        "//pkg/apis/batch/v2alpha1:go_default_library",
        "//pkg/apis/certificates/v1beta1:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//pkg/apis/networking/v1:go_default_library",
        "//pkg/apis/policy/v1beta1:go_default_library",
        "//pkg/apis/rbac/v1alpha1:go_default_library",
        "//pkg/apis/rbac/v1beta1:go_default_library",
        "//pkg/apis/settings/v1alpha1:go_default_library",
        "//pkg/apis/storage/v1:go_default_library",
        "//pkg/apis/storage/v1beta1:go_default_library",
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
