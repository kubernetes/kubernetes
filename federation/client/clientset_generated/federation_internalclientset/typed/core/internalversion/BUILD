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
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/watch:go_default_library",
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
        "//federation/client/clientset_generated/federation_internalclientset/typed/core/internalversion/fake:all-srcs",
    ],
    tags = ["automanaged"],
)
