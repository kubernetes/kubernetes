package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "factory.go",
        "generic.go",
    ],
    deps = [
        "//vendor/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions:go_default_library",
        "//vendor/k8s.io/apiextensions-apiserver/pkg/client/clientset/internalclientset:go_default_library",
        "//vendor/k8s.io/apiextensions-apiserver/pkg/client/informers/internalversion/apiextensions:go_default_library",
        "//vendor/k8s.io/apiextensions-apiserver/pkg/client/informers/internalversion/internalinterfaces:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/client-go/tools/cache:go_default_library",
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
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/client/informers/internalversion/apiextensions:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/client/informers/internalversion/internalinterfaces:all-srcs",
    ],
    tags = ["automanaged"],
)
