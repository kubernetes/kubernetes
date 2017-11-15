load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "controllerrevision.go",
        "daemonset.go",
        "deployment.go",
        "interface.go",
        "replicaset.go",
        "statefulset.go",
    ],
    importpath = "k8s.io/client-go/informers/apps/v1beta2",
    visibility = ["//visibility:public"],
    deps = [
        "//informers/internalinterfaces:go_default_library",
        "//kubernetes:go_default_library",
        "//listers/apps/v1beta2:go_default_library",
        "//tools/cache:go_default_library",
        "//vendor/k8s.io/api/apps/v1beta2:go_default_library",
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
