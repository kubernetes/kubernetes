package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
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
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/apps/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/apps/v1beta1/fake:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/authentication/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/authentication/v1beta1/fake:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/authorization/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/authorization/v1beta1/fake:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/autoscaling/v1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/autoscaling/v1/fake:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/batch/v1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/batch/v1/fake:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/batch/v2alpha1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/batch/v2alpha1/fake:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/certificates/v1alpha1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/certificates/v1alpha1/fake:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/core/v1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/core/v1/fake:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/extensions/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/extensions/v1beta1/fake:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/policy/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/policy/v1beta1/fake:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/rbac/v1alpha1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/rbac/v1alpha1/fake:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/storage/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/storage/v1beta1/fake:go_default_library",
        "//pkg/client/testing/core:go_default_library",
        "//pkg/client/typed/discovery:go_default_library",
        "//pkg/client/typed/discovery/fake:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/watch",
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
)
