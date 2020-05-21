load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "factory.go",
        "generic.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/openshift/client-go/apps/informers/externalversions",
    importpath = "github.com/openshift/client-go/apps/informers/externalversions",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/github.com/openshift/api/apps/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/apps/clientset/versioned:go_default_library",
        "//vendor/github.com/openshift/client-go/apps/informers/externalversions/apps:go_default_library",
        "//vendor/github.com/openshift/client-go/apps/informers/externalversions/internalinterfaces:go_default_library",
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
        "//vendor/github.com/openshift/client-go/apps/informers/externalversions/apps:all-srcs",
        "//vendor/github.com/openshift/client-go/apps/informers/externalversions/internalinterfaces:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
