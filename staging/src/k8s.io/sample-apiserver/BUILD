package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "sample-apiserver",
    embed = [":go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = ["main.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/sample-apiserver",
    importpath = "k8s.io/sample-apiserver",
    deps = [
        "//staging/src/k8s.io/apiserver/pkg/server:go_default_library",
        "//staging/src/k8s.io/component-base/logs:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/cmd/server:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//staging/src/k8s.io/sample-apiserver/pkg/admission/plugin/banflunder:all-srcs",
        "//staging/src/k8s.io/sample-apiserver/pkg/admission/wardleinitializer:all-srcs",
        "//staging/src/k8s.io/sample-apiserver/pkg/apis/wardle:all-srcs",
        "//staging/src/k8s.io/sample-apiserver/pkg/apiserver:all-srcs",
        "//staging/src/k8s.io/sample-apiserver/pkg/cmd/server:all-srcs",
        "//staging/src/k8s.io/sample-apiserver/pkg/generated/clientset/versioned:all-srcs",
        "//staging/src/k8s.io/sample-apiserver/pkg/generated/informers/externalversions:all-srcs",
        "//staging/src/k8s.io/sample-apiserver/pkg/generated/listers/wardle/v1alpha1:all-srcs",
        "//staging/src/k8s.io/sample-apiserver/pkg/generated/listers/wardle/v1beta1:all-srcs",
        "//staging/src/k8s.io/sample-apiserver/pkg/registry:all-srcs",
    ],
    tags = ["automanaged"],
)
