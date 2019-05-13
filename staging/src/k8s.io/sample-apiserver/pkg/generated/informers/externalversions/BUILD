load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "factory.go",
        "generic.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/sample-apiserver/pkg/generated/informers/externalversions",
    importpath = "k8s.io/sample-apiserver/pkg/generated/informers/externalversions",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/apis/wardle/v1alpha1:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/apis/wardle/v1beta1:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/generated/clientset/versioned:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/generated/informers/externalversions/internalinterfaces:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/generated/informers/externalversions/wardle:go_default_library",
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
        "//staging/src/k8s.io/sample-apiserver/pkg/generated/informers/externalversions/internalinterfaces:all-srcs",
        "//staging/src/k8s.io/sample-apiserver/pkg/generated/informers/externalversions/wardle:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
