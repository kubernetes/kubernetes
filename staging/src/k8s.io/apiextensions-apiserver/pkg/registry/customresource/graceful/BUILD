load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["graceful_shutdown.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiextensions-apiserver/pkg/registry/customresource/graceful",
    importpath = "k8s.io/apiextensions-apiserver/pkg/registry/customresource/graceful",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/waitgroup:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/registry/generic:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/storagebackend:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/storagebackend/factory:go_default_library",
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
