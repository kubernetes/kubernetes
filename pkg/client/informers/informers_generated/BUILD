package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "factory.go",
        "generic.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/apps:go_default_library",
        "//pkg/apis/apps/v1beta1:go_default_library",
        "//pkg/apis/autoscaling:go_default_library",
        "//pkg/apis/autoscaling/v1:go_default_library",
        "//pkg/apis/batch:go_default_library",
        "//pkg/apis/batch/v1:go_default_library",
        "//pkg/apis/batch/v2alpha1:go_default_library",
        "//pkg/apis/certificates:go_default_library",
        "//pkg/apis/certificates/v1alpha1:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//pkg/apis/policy:go_default_library",
        "//pkg/apis/policy/v1beta1:go_default_library",
        "//pkg/apis/rbac:go_default_library",
        "//pkg/apis/rbac/v1alpha1:go_default_library",
        "//pkg/apis/storage:go_default_library",
        "//pkg/apis/storage/v1beta1:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/clientset_generated/release_1_5:go_default_library",
        "//pkg/client/informers/informers_generated/apps:go_default_library",
        "//pkg/client/informers/informers_generated/autoscaling:go_default_library",
        "//pkg/client/informers/informers_generated/batch:go_default_library",
        "//pkg/client/informers/informers_generated/certificates:go_default_library",
        "//pkg/client/informers/informers_generated/core:go_default_library",
        "//pkg/client/informers/informers_generated/extensions:go_default_library",
        "//pkg/client/informers/informers_generated/internalinterfaces:go_default_library",
        "//pkg/client/informers/informers_generated/policy:go_default_library",
        "//pkg/client/informers/informers_generated/rbac:go_default_library",
        "//pkg/client/informers/informers_generated/storage:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
    ],
)
