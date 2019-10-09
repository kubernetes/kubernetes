load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["rate_limiter.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/component-base/metrics/prometheus/ratelimiter",
    importpath = "k8s.io/component-base/metrics/prometheus/ratelimiter",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/client-go/util/flowcontrol:go_default_library",
        "//staging/src/k8s.io/component-base/metrics:go_default_library",
        "//staging/src/k8s.io/component-base/metrics/legacyregistry:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["rate_limiter_test.go"],
    embed = [":go_default_library"],
    deps = ["//staging/src/k8s.io/client-go/util/flowcontrol:go_default_library"],
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
