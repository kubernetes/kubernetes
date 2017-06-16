package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "admissionregistration_client.go",
        "doc.go",
        "externaladmissionhookconfiguration.go",
        "generated_expansion.go",
        "initializerconfiguration.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/admissionregistration/v1alpha1:go_default_library",
        "//pkg/client/clientset_generated/clientset/scheme:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//pkg/client/clientset_generated/clientset/typed/admissionregistration/v1alpha1/fake:all-srcs",
    ],
    tags = ["automanaged"],
)
