load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "csidriver.go",
        "csinodeinfo.go",
        "interface.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/csi-api/pkg/client/informers/externalversions/csi/v1alpha1",
    importpath = "k8s.io/csi-api/pkg/client/informers/externalversions/csi/v1alpha1",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//staging/src/k8s.io/csi-api/pkg/apis/csi/v1alpha1:go_default_library",
        "//staging/src/k8s.io/csi-api/pkg/client/clientset/versioned:go_default_library",
        "//staging/src/k8s.io/csi-api/pkg/client/informers/externalversions/internalinterfaces:go_default_library",
        "//staging/src/k8s.io/csi-api/pkg/client/listers/csi/v1alpha1:go_default_library",
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
