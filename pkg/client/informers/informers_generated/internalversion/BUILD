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
        "//pkg/api:go_default_library",
        "//pkg/apis/admissionregistration:go_default_library",
        "//pkg/apis/apps:go_default_library",
        "//pkg/apis/autoscaling:go_default_library",
        "//pkg/apis/batch:go_default_library",
        "//pkg/apis/certificates:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apis/networking:go_default_library",
        "//pkg/apis/policy:go_default_library",
        "//pkg/apis/rbac:go_default_library",
        "//pkg/apis/settings:go_default_library",
        "//pkg/apis/storage:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/informers/informers_generated/internalversion/admissionregistration:go_default_library",
        "//pkg/client/informers/informers_generated/internalversion/apps:go_default_library",
        "//pkg/client/informers/informers_generated/internalversion/autoscaling:go_default_library",
        "//pkg/client/informers/informers_generated/internalversion/batch:go_default_library",
        "//pkg/client/informers/informers_generated/internalversion/certificates:go_default_library",
        "//pkg/client/informers/informers_generated/internalversion/core:go_default_library",
        "//pkg/client/informers/informers_generated/internalversion/extensions:go_default_library",
        "//pkg/client/informers/informers_generated/internalversion/internalinterfaces:go_default_library",
        "//pkg/client/informers/informers_generated/internalversion/networking:go_default_library",
        "//pkg/client/informers/informers_generated/internalversion/policy:go_default_library",
        "//pkg/client/informers/informers_generated/internalversion/rbac:go_default_library",
        "//pkg/client/informers/informers_generated/internalversion/settings:go_default_library",
        "//pkg/client/informers/informers_generated/internalversion/storage:go_default_library",
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
        "//pkg/client/informers/informers_generated/internalversion/admissionregistration:all-srcs",
        "//pkg/client/informers/informers_generated/internalversion/apps:all-srcs",
        "//pkg/client/informers/informers_generated/internalversion/autoscaling:all-srcs",
        "//pkg/client/informers/informers_generated/internalversion/batch:all-srcs",
        "//pkg/client/informers/informers_generated/internalversion/certificates:all-srcs",
        "//pkg/client/informers/informers_generated/internalversion/core:all-srcs",
        "//pkg/client/informers/informers_generated/internalversion/extensions:all-srcs",
        "//pkg/client/informers/informers_generated/internalversion/internalinterfaces:all-srcs",
        "//pkg/client/informers/informers_generated/internalversion/networking:all-srcs",
        "//pkg/client/informers/informers_generated/internalversion/policy:all-srcs",
        "//pkg/client/informers/informers_generated/internalversion/rbac:all-srcs",
        "//pkg/client/informers/informers_generated/internalversion/settings:all-srcs",
        "//pkg/client/informers/informers_generated/internalversion/storage:all-srcs",
    ],
    tags = ["automanaged"],
)
