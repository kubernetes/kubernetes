package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "batch.go",
        "core.go",
        "extensions.go",
        "factory.go",
        "generic.go",
        "rbac.go",
        "storage.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/legacylisters:go_default_library",
        "//pkg/client/listers/batch/v1:go_default_library",
        "//pkg/client/listers/core/internalversion:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/apis/pkg/api",
        "//vendor:k8s.io/apis/pkg/api/v1",
        "//vendor:k8s.io/apis/pkg/apis/batch",
        "//vendor:k8s.io/apis/pkg/apis/batch/v1",
        "//vendor:k8s.io/apis/pkg/apis/extensions",
        "//vendor:k8s.io/apis/pkg/apis/extensions/v1beta1",
        "//vendor:k8s.io/apis/pkg/apis/rbac",
        "//vendor:k8s.io/apis/pkg/apis/storage/v1beta1",
        "//vendor:k8s.io/client-go/tools/cache",
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
