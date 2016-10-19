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
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/apps/unversioned:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/apps/unversioned/fake:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/authentication/unversioned:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/authentication/unversioned/fake:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/authorization/unversioned:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/authorization/unversioned/fake:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/autoscaling/unversioned:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/autoscaling/unversioned/fake:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/batch/unversioned:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/batch/unversioned/fake:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/certificates/unversioned:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/certificates/unversioned/fake:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/core/unversioned:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/core/unversioned/fake:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/extensions/unversioned:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/extensions/unversioned/fake:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/policy/unversioned:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/policy/unversioned/fake:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/rbac/unversioned:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/rbac/unversioned/fake:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/storage/unversioned:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/storage/unversioned/fake:go_default_library",
        "//pkg/client/testing/core:go_default_library",
        "//pkg/client/typed/discovery:go_default_library",
        "//pkg/client/typed/discovery/fake:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/watch:go_default_library",
    ],
)
