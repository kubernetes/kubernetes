package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "clusterrole.go",
        "clusterrolebinding.go",
        "interface.go",
        "role.go",
        "rolebinding.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//vendor/k8s.io/client-go/informers/internalinterfaces:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/k8s.io/client-go/listers/rbac/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/rbac/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/tools/cache:go_default_library",
    ],
)
