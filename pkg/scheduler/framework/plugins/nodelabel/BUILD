load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["node_label.go"],
    importpath = "k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodelabel",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/scheduler/apis/config:go_default_library",
        "//pkg/scheduler/apis/config/validation:go_default_library",
        "//pkg/scheduler/framework/v1alpha1:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["node_label_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/scheduler/apis/config:go_default_library",
        "//pkg/scheduler/framework/v1alpha1:go_default_library",
        "//pkg/scheduler/internal/cache:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
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
