load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "install.go",
        "scheme.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubectl/scheme",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/k8s.io/api/admission/install:go_default_library",
        "//vendor/k8s.io/api/admissionregistration/install:go_default_library",
        "//vendor/k8s.io/api/apps/install:go_default_library",
        "//vendor/k8s.io/api/authentication/install:go_default_library",
        "//vendor/k8s.io/api/authorization/install:go_default_library",
        "//vendor/k8s.io/api/autoscaling/install:go_default_library",
        "//vendor/k8s.io/api/batch/install:go_default_library",
        "//vendor/k8s.io/api/certificates/install:go_default_library",
        "//vendor/k8s.io/api/core/install:go_default_library",
        "//vendor/k8s.io/api/extensions/install:go_default_library",
        "//vendor/k8s.io/api/imagepolicy/install:go_default_library",
        "//vendor/k8s.io/api/networking/install:go_default_library",
        "//vendor/k8s.io/api/policy/install:go_default_library",
        "//vendor/k8s.io/api/rbac/install:go_default_library",
        "//vendor/k8s.io/api/scheduling/install:go_default_library",
        "//vendor/k8s.io/api/settings/install:go_default_library",
        "//vendor/k8s.io/api/storage/install:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apimachinery/announced:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apimachinery/registered:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
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
    visibility = ["//visibility:public"],
)
