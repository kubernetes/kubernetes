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
        "fake_apiregistration_client.go",
        "fake_apiservice.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//cmd/kube-aggregator/pkg/apis/apiregistration:go_default_library",
        "//cmd/kube-aggregator/pkg/client/clientset_generated/internalclientset/typed/apiregistration/internalversion:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/testing/core:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/watch",
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
