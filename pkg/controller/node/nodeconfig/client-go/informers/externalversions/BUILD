load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "factory.go",
        "generic.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/controller/node/nodeconfig/client-go/informers/externalversions",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/controller/node/nodeconfig/client-go/clientset/versioned:go_default_library",
        "//pkg/controller/node/nodeconfig/client-go/informers/externalversions/internalinterfaces:go_default_library",
        "//pkg/controller/node/nodeconfig/client-go/informers/externalversions/nodeconfig:go_default_library",
        "//pkg/controller/node/nodeconfig/v1alpha1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//pkg/controller/node/nodeconfig/client-go/informers/externalversions/internalinterfaces:all-srcs",
        "//pkg/controller/node/nodeconfig/client-go/informers/externalversions/nodeconfig:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
