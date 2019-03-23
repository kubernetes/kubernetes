package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "cauldron",
    embed = [":go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = ["cauldron.go"],
    importpath = "k8s.io/kubernetes/test/soak/cauldron",
    deps = [
        "//pkg/apis/core:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/intstr:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//test/e2e/framework:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
