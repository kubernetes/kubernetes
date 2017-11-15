load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "daemonset.go",
        "deployment.go",
        "ingress.go",
        "interface.go",
        "podsecuritypolicy.go",
        "replicaset.go",
    ],
    importpath = "k8s.io/client-go/informers/extensions/v1beta1",
    visibility = ["//visibility:public"],
    deps = [
        "//informers/internalinterfaces:go_default_library",
        "//kubernetes:go_default_library",
        "//listers/extensions/v1beta1:go_default_library",
        "//tools/cache:go_default_library",
        "//vendor/k8s.io/api/extensions/v1beta1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
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
