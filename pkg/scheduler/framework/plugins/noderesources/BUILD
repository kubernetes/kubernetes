load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "balanced_allocation.go",
        "fit.go",
        "least_allocated.go",
        "most_allocated.go",
        "test_util.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/scheduler/algorithm/predicates:go_default_library",
        "//pkg/scheduler/algorithm/priorities:go_default_library",
        "//pkg/scheduler/framework/plugins/migration:go_default_library",
        "//pkg/scheduler/framework/v1alpha1:go_default_library",
        "//pkg/scheduler/nodeinfo:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
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

go_test(
    name = "go_default_test",
    srcs = [
        "balanced_allocation_test.go",
        "fit_test.go",
        "least_allocated_test.go",
        "most_allocated_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/apis/core/v1/helper:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/scheduler/algorithm/predicates:go_default_library",
        "//pkg/scheduler/framework/plugins/migration:go_default_library",
        "//pkg/scheduler/framework/v1alpha1:go_default_library",
        "//pkg/scheduler/nodeinfo:go_default_library",
        "//pkg/scheduler/nodeinfo/snapshot:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/component-base/featuregate/testing:go_default_library",
    ],
)
