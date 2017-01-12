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
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/apps/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/apps/internalversion/fake:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/authentication/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/authentication/internalversion/fake:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/authorization/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/authorization/internalversion/fake:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/autoscaling/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/autoscaling/internalversion/fake:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/batch/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/batch/internalversion/fake:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/certificates/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/certificates/internalversion/fake:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/core/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/core/internalversion/fake:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/extensions/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/extensions/internalversion/fake:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/policy/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/policy/internalversion/fake:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion/fake:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/storage/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/storage/internalversion/fake:go_default_library",
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
