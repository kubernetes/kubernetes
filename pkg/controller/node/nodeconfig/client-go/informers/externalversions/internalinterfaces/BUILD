load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["factory_interfaces.go"],
    importpath = "k8s.io/kubernetes/pkg/controller/node/nodeconfig/client-go/informers/externalversions/internalinterfaces",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/controller/node/nodeconfig/client-go/clientset/versioned:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
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
