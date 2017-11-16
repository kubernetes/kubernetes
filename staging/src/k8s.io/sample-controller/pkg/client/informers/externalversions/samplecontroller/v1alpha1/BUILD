load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "foo.go",
        "interface.go",
    ],
    importpath = "k8s.io/sample-controller/pkg/client/informers/externalversions/samplecontroller/v1alpha1",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//vendor/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/k8s.io/sample-controller/pkg/apis/samplecontroller/v1alpha1:go_default_library",
        "//vendor/k8s.io/sample-controller/pkg/client/clientset/versioned:go_default_library",
        "//vendor/k8s.io/sample-controller/pkg/client/informers/externalversions/internalinterfaces:go_default_library",
        "//vendor/k8s.io/sample-controller/pkg/client/listers/samplecontroller/v1alpha1:go_default_library",
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
