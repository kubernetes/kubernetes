package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "clientset_generated.go",
        "doc.go",
        "register.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//federation/apis/federation/v1beta1:go_default_library",
        "//federation/client/clientset_generated/federation_clientset:go_default_library",
        "//federation/client/clientset_generated/federation_clientset/typed/autoscaling/v1:go_default_library",
        "//federation/client/clientset_generated/federation_clientset/typed/autoscaling/v1/fake:go_default_library",
        "//federation/client/clientset_generated/federation_clientset/typed/batch/v1:go_default_library",
        "//federation/client/clientset_generated/federation_clientset/typed/batch/v1/fake:go_default_library",
        "//federation/client/clientset_generated/federation_clientset/typed/core/v1:go_default_library",
        "//federation/client/clientset_generated/federation_clientset/typed/core/v1/fake:go_default_library",
        "//federation/client/clientset_generated/federation_clientset/typed/extensions/v1beta1:go_default_library",
        "//federation/client/clientset_generated/federation_clientset/typed/extensions/v1beta1/fake:go_default_library",
        "//federation/client/clientset_generated/federation_clientset/typed/federation/v1beta1:go_default_library",
        "//federation/client/clientset_generated/federation_clientset/typed/federation/v1beta1/fake:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/autoscaling/v1:go_default_library",
        "//pkg/apis/batch/v1:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//vendor/k8s.io/client-go/discovery:go_default_library",
        "//vendor/k8s.io/client-go/discovery/fake:go_default_library",
        "//vendor/k8s.io/client-go/testing:go_default_library",
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
)
