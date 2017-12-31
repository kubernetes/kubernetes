load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["nodeconfig_controller.go"],
    importpath = "k8s.io/kubernetes/pkg/controller/nodeconfig",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/controller:go_default_library",
        "//pkg/controller/node/nodeconfig/client-go/clientset/versioned:go_default_library",
        "//pkg/controller/node/nodeconfig/client-go/informers/externalversions:go_default_library",
        "//pkg/controller/node/nodeconfig/client-go/listers/nodeconfig/v1alpha1:go_default_library",
        "//pkg/controller/node/nodeconfig/v1alpha1:go_default_library",
        "//pkg/controller/node/util:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/client-go/informers/core/v1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/k8s.io/client-go/listers/core/v1:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
        "//vendor/k8s.io/client-go/tools/cache:go_default_library",
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
