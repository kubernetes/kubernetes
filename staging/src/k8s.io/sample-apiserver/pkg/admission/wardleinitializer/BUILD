package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "interfaces.go",
        "wardleinitializer.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/k8s.io/apiserver/pkg/admission:go_default_library",
        "//vendor/k8s.io/sample-apiserver/pkg/client/informers_generated/internalversion:go_default_library",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = ["wardleinitializer_test.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor/k8s.io/apiserver/pkg/admission:go_default_library",
        "//vendor/k8s.io/sample-apiserver/pkg/admission/wardleinitializer:go_default_library",
        "//vendor/k8s.io/sample-apiserver/pkg/client/clientset_generated/internalclientset/fake:go_default_library",
        "//vendor/k8s.io/sample-apiserver/pkg/client/informers_generated/internalversion:go_default_library",
    ],
)
