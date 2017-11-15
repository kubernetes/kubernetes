load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "interface.go",
        "podpreset.go",
    ],
    importpath = "k8s.io/client-go/informers/settings/v1alpha1",
    visibility = ["//visibility:public"],
    deps = [
        "//informers/internalinterfaces:go_default_library",
        "//kubernetes:go_default_library",
        "//listers/settings/v1alpha1:go_default_library",
        "//tools/cache:go_default_library",
        "//vendor/k8s.io/api/settings/v1alpha1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
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
