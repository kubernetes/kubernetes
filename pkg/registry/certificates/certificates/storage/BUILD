package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["storage.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/apis/certificates:go_default_library",
        "//pkg/registry/cachesize:go_default_library",
        "//pkg/registry/certificates/certificates:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apiserver/pkg/endpoints/request",
        "//vendor:k8s.io/apiserver/pkg/registry/generic",
        "//vendor:k8s.io/apiserver/pkg/registry/generic/registry",
        "//vendor:k8s.io/apiserver/pkg/registry/rest",
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
