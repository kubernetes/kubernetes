package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "matcher.go",
        "options.go",
        "storage_decorator.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/storage:go_default_library",
        "//pkg/storage/storagebackend:go_default_library",
        "//pkg/storage/storagebackend/factory:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
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
        "//pkg/genericapiserver/registry/generic/registry:all-srcs",
        "//pkg/genericapiserver/registry/generic/rest:all-srcs",
    ],
    tags = ["automanaged"],
)
