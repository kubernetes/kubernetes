package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "configmap.go",
        "core_client.go",
        "doc.go",
        "event.go",
        "generated_expansion.go",
        "namespace.go",
        "namespace_expansion.go",
        "secret.go",
        "service.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apimachinery/pkg/apis/meta/v1"],
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
        "//federation/client/clientset_generated/federation_clientset/typed/core/v1/fake:all-srcs",
    ],
    tags = ["automanaged"],
)
