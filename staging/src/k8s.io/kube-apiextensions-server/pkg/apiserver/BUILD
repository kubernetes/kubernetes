package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["apiserver.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/apimachinery/announced:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apimachinery/registered:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/version:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/registry/rest:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/server:go_default_library",
        "//vendor/k8s.io/kube-apiextensions-server/pkg/apis/apiextensions:go_default_library",
        "//vendor/k8s.io/kube-apiextensions-server/pkg/apis/apiextensions/install:go_default_library",
        "//vendor/k8s.io/kube-apiextensions-server/pkg/apis/apiextensions/v1alpha1:go_default_library",
        "//vendor/k8s.io/kube-apiextensions-server/pkg/client/clientset/clientset:go_default_library",
        "//vendor/k8s.io/kube-apiextensions-server/pkg/client/clientset/internalclientset:go_default_library",
        "//vendor/k8s.io/kube-apiextensions-server/pkg/client/informers/externalversions:go_default_library",
        "//vendor/k8s.io/kube-apiextensions-server/pkg/client/informers/internalversion:go_default_library",
        "//vendor/k8s.io/kube-apiextensions-server/pkg/registry/customresource:go_default_library",
    ],
)
