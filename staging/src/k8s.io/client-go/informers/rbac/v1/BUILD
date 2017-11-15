load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "clusterrole.go",
        "clusterrolebinding.go",
        "interface.go",
        "role.go",
        "rolebinding.go",
    ],
    importpath = "k8s.io/client-go/informers/rbac/v1",
    visibility = ["//visibility:public"],
    deps = [
        "//informers/internalinterfaces:go_default_library",
        "//kubernetes:go_default_library",
        "//listers/rbac/v1:go_default_library",
        "//tools/cache:go_default_library",
        "//vendor/k8s.io/api/rbac/v1:go_default_library",
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
