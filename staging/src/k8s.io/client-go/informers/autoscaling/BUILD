load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["interface.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/client-go/informers/autoscaling",
    importpath = "k8s.io/client-go/informers/autoscaling",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/client-go/informers/autoscaling/v1:go_default_library",
        "//staging/src/k8s.io/client-go/informers/autoscaling/v2beta1:go_default_library",
        "//staging/src/k8s.io/client-go/informers/autoscaling/v2beta2:go_default_library",
        "//staging/src/k8s.io/client-go/informers/internalinterfaces:go_default_library",
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
        "//staging/src/k8s.io/client-go/informers/autoscaling/v1:all-srcs",
        "//staging/src/k8s.io/client-go/informers/autoscaling/v2beta1:all-srcs",
        "//staging/src/k8s.io/client-go/informers/autoscaling/v2beta2:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
