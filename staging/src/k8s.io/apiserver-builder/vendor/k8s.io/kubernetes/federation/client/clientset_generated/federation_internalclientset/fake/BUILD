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
        "//federation/apis/federation/install:go_default_library",
        "//federation/client/clientset_generated/federation_internalclientset:go_default_library",
        "//federation/client/clientset_generated/federation_internalclientset/typed/autoscaling/internalversion:go_default_library",
        "//federation/client/clientset_generated/federation_internalclientset/typed/autoscaling/internalversion/fake:go_default_library",
        "//federation/client/clientset_generated/federation_internalclientset/typed/batch/internalversion:go_default_library",
        "//federation/client/clientset_generated/federation_internalclientset/typed/batch/internalversion/fake:go_default_library",
        "//federation/client/clientset_generated/federation_internalclientset/typed/core/internalversion:go_default_library",
        "//federation/client/clientset_generated/federation_internalclientset/typed/core/internalversion/fake:go_default_library",
        "//federation/client/clientset_generated/federation_internalclientset/typed/extensions/internalversion:go_default_library",
        "//federation/client/clientset_generated/federation_internalclientset/typed/extensions/internalversion/fake:go_default_library",
        "//federation/client/clientset_generated/federation_internalclientset/typed/federation/internalversion:go_default_library",
        "//federation/client/clientset_generated/federation_internalclientset/typed/federation/internalversion/fake:go_default_library",
        "//pkg/api/install:go_default_library",
        "//pkg/apis/autoscaling/install:go_default_library",
        "//pkg/apis/batch/install:go_default_library",
        "//pkg/apis/extensions/install:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/announced",
        "//vendor:k8s.io/apimachinery/pkg/apimachinery/registered",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/discovery",
        "//vendor:k8s.io/client-go/discovery/fake",
        "//vendor:k8s.io/client-go/testing",
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
