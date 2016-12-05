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
        "batch.go",
        "core.go",
        "extensions.go",
        "factory.go",
        "generic.go",
        "rbac.go",
        "storage.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/batch:go_default_library",
        "//pkg/apis/batch/v1:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//pkg/apis/rbac:go_default_library",
        "//pkg/apis/storage/v1beta1:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/clientset_generated/release_1_5:go_default_library",
        "//pkg/client/listers/batch/v1:go_default_library",
        "//pkg/client/listers/core/internalversion:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/watch:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)
