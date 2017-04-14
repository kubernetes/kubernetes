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
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//vendor/k8s.io/client-go/pkg/api/v1:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/apps/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/authentication/v1:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/authentication/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/authorization/v1:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/authorization/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/autoscaling/v1:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/autoscaling/v2alpha1:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/batch/v1:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/batch/v2alpha1:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/certificates/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/extensions/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/policy/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/rbac/v1alpha1:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/rbac/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/settings/v1alpha1:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/storage/v1:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/storage/v1beta1:go_default_library",
    ],
)
