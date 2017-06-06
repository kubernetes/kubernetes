package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "register.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/admissionregistration/v1alpha1:go_default_library",
        "//pkg/apis/apps/v1beta1:go_default_library",
        "//pkg/apis/authentication/v1:go_default_library",
        "//pkg/apis/authentication/v1beta1:go_default_library",
        "//pkg/apis/authorization/v1:go_default_library",
        "//pkg/apis/authorization/v1beta1:go_default_library",
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
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
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
