package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["helpers_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/client-go/pkg/api/v1:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = ["helpers.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/selection:go_default_library",
        "//vendor/k8s.io/client-go/pkg/api:go_default_library",
        "//vendor/k8s.io/client-go/pkg/api/helper:go_default_library",
        "//vendor/k8s.io/client-go/pkg/api/v1:go_default_library",
    ],
)
