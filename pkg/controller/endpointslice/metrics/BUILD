load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "cache.go",
        "metrics.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/controller/endpointslice/metrics",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/controller/util/endpoint:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/component-base/metrics:go_default_library",
        "//staging/src/k8s.io/component-base/metrics/legacyregistry:go_default_library",
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
    srcs = ["cache_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/controller/util/endpoint:go_default_library",
        "//staging/src/k8s.io/api/discovery/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
    ],
)
