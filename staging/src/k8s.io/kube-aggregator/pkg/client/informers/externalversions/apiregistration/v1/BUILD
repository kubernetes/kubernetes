load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "apiservice.go",
        "interface.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kube-aggregator/pkg/client/informers/externalversions/apiregistration/v1",
    importpath = "k8s.io/kube-aggregator/pkg/client/informers/externalversions/apiregistration/v1",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/apis/apiregistration/v1:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/client/informers/externalversions/internalinterfaces:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/client/listers/apiregistration/v1:go_default_library",
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
    visibility = ["//visibility:public"],
)
