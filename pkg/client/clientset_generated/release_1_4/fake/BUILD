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
        "clientset_generated.go",
        "doc.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/client/clientset_generated/release_1_4:go_default_library",
        "//pkg/client/clientset_generated/release_1_4/typed/apps/v1alpha1:go_default_library",
        "//pkg/client/clientset_generated/release_1_4/typed/apps/v1alpha1/fake:go_default_library",
        "//pkg/client/clientset_generated/release_1_4/typed/authentication/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/release_1_4/typed/authentication/v1beta1/fake:go_default_library",
        "//pkg/client/clientset_generated/release_1_4/typed/authorization/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/release_1_4/typed/authorization/v1beta1/fake:go_default_library",
        "//pkg/client/clientset_generated/release_1_4/typed/autoscaling/v1:go_default_library",
        "//pkg/client/clientset_generated/release_1_4/typed/autoscaling/v1/fake:go_default_library",
        "//pkg/client/clientset_generated/release_1_4/typed/batch/v1:go_default_library",
        "//pkg/client/clientset_generated/release_1_4/typed/batch/v1/fake:go_default_library",
        "//pkg/client/clientset_generated/release_1_4/typed/certificates/v1alpha1:go_default_library",
        "//pkg/client/clientset_generated/release_1_4/typed/certificates/v1alpha1/fake:go_default_library",
        "//pkg/client/clientset_generated/release_1_4/typed/core/v1:go_default_library",
        "//pkg/client/clientset_generated/release_1_4/typed/core/v1/fake:go_default_library",
        "//pkg/client/clientset_generated/release_1_4/typed/extensions/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/release_1_4/typed/extensions/v1beta1/fake:go_default_library",
        "//pkg/client/clientset_generated/release_1_4/typed/policy/v1alpha1:go_default_library",
        "//pkg/client/clientset_generated/release_1_4/typed/policy/v1alpha1/fake:go_default_library",
        "//pkg/client/clientset_generated/release_1_4/typed/rbac/v1alpha1:go_default_library",
        "//pkg/client/clientset_generated/release_1_4/typed/rbac/v1alpha1/fake:go_default_library",
        "//pkg/client/clientset_generated/release_1_4/typed/storage/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/release_1_4/typed/storage/v1beta1/fake:go_default_library",
        "//pkg/client/testing/core:go_default_library",
        "//pkg/client/typed/discovery:go_default_library",
        "//pkg/client/typed/discovery/fake:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/watch:go_default_library",
    ],
)
