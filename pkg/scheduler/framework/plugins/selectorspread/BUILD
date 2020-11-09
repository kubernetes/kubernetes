load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["selector_spread.go"],
    importpath = "k8s.io/kubernetes/pkg/scheduler/framework/plugins/selectorspread",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/scheduler/framework:go_default_library",
        "//pkg/scheduler/framework/plugins/helper:go_default_library",
        "//pkg/util/node:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/client-go/listers/apps/v1:go_default_library",
        "//staging/src/k8s.io/client-go/listers/core/v1:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "selector_spread_perf_test.go",
        "selector_spread_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/scheduler/framework:go_default_library",
        "//pkg/scheduler/framework/runtime:go_default_library",
        "//pkg/scheduler/internal/cache:go_default_library",
        "//pkg/scheduler/internal/parallelize:go_default_library",
        "//pkg/scheduler/testing:go_default_library",
        "//staging/src/k8s.io/api/apps/v1:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/client-go/informers:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/fake:go_default_library",
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
