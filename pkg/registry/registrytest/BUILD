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
        "endpoint.go",
        "etcd.go",
        "node.go",
        "service.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/rest/resttest:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/fields:go_default_library",
        "//pkg/registry/generic/registry:go_default_library",
        "//pkg/storage/etcd:go_default_library",
        "//pkg/storage/etcd/testing:go_default_library",
        "//pkg/storage/storagebackend:go_default_library",
        "//pkg/storage/testing:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/apiserver/pkg/request",
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
