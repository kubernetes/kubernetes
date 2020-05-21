load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "build.go",
        "buildconfig.go",
        "interface.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/openshift/client-go/build/informers/externalversions/build/v1",
    importpath = "github.com/openshift/client-go/build/informers/externalversions/build/v1",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/github.com/openshift/api/build/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/build/clientset/versioned:go_default_library",
        "//vendor/github.com/openshift/client-go/build/informers/externalversions/internalinterfaces:go_default_library",
        "//vendor/github.com/openshift/client-go/build/listers/build/v1:go_default_library",
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
