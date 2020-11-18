package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "fake_extender.go",
        "fake_plugins.go",
        "framework_helpers.go",
        "workload_prep.go",
        "wrappers.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/scheduler/testing",
    deps = [
        "//pkg/scheduler/apis/config:go_default_library",
        "//pkg/scheduler/framework:go_default_library",
        "//pkg/scheduler/framework/runtime:go_default_library",
        "//pkg/scheduler/util:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/component-helpers/scheduling/corev1:go_default_library",
        "//staging/src/k8s.io/kube-scheduler/extender/v1:go_default_library",
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
)
