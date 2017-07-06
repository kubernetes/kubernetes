package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "fischer.go",
        "flunder.go",
        "interface.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//vendor/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/k8s.io/sample-apiserver/pkg/apis/wardle:go_default_library",
        "//vendor/k8s.io/sample-apiserver/pkg/client/clientset_generated/internalclientset:go_default_library",
        "//vendor/k8s.io/sample-apiserver/pkg/client/informers_generated/internalversion/internalinterfaces:go_default_library",
        "//vendor/k8s.io/sample-apiserver/pkg/client/listers_generated/wardle/internalversion:go_default_library",
    ],
)
