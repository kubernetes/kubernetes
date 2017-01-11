package(default_visibility = ["//visibility:public"])

licenses(["notice"])

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
    tags = ["automanaged"],
    deps = [
        "//cmd/kube-aggregator/pkg/apis/apiregistration:go_default_library",
        "//cmd/kube-aggregator/pkg/apis/apiregistration/v1alpha1:go_default_library",
        "//cmd/kube-aggregator/pkg/client/clientset_generated/clientset:go_default_library",
        "//cmd/kube-aggregator/pkg/client/clientset_generated/internalclientset:go_default_library",
        "//cmd/kube-aggregator/pkg/client/informers/apiregistration:go_default_library",
        "//cmd/kube-aggregator/pkg/client/informers/internalinterfaces:go_default_library",
        "//pkg/client/cache:go_default_library",
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
        "//cmd/kube-aggregator/pkg/client/informers/apiregistration:all-srcs",
        "//cmd/kube-aggregator/pkg/client/informers/internalinterfaces:all-srcs",
    ],
    tags = ["automanaged"],
)
