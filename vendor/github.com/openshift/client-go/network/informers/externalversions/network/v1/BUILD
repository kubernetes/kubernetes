load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "clusternetwork.go",
        "egressnetworkpolicy.go",
        "hostsubnet.go",
        "interface.go",
        "netnamespace.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/openshift/client-go/network/informers/externalversions/network/v1",
    importpath = "github.com/openshift/client-go/network/informers/externalversions/network/v1",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/github.com/openshift/api/network/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/network/clientset/versioned:go_default_library",
        "//vendor/github.com/openshift/client-go/network/informers/externalversions/internalinterfaces:go_default_library",
        "//vendor/github.com/openshift/client-go/network/listers/network/v1:go_default_library",
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
