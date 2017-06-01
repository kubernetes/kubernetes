package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["install.go"],
    tags = ["automanaged"],
    visibility = [
        "//pkg/client/tests:__subpackages__",
        "//pkg/controller/podautoscaler/metrics:__subpackages__",
        "//vendor/k8s.io/apiserver/pkg/endpoints/handlers:__subpackages__",
        "//vendor/k8s.io/apiserver/pkg/storage/tests:__subpackages__",
        "//vendor/k8s.io/client-go/kubernetes/fake:__subpackages__",
    ],
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/apimachinery/announced:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apimachinery/registered:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/client-go/pkg/api:go_default_library",
        "//vendor/k8s.io/client-go/pkg/api/v1:go_default_library",
    ],
)
