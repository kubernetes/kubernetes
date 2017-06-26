package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "key_store.go",
        "kms_factory.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//cmd/kube-apiserver/app/options:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/kubeapiserver/options:go_default_library",
        "//pkg/registry/core/configmap:go_default_library",
        "//pkg/registry/core/configmap/storage:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/endpoints/request:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/registry/generic:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/server/storage:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/storagebackend:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/value:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/kms:go_default_library",
    ],
)
