load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["sampling-histogram.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/component-base/metrics/prometheusextension",
    importpath = "k8s.io/component-base/metrics/prometheusextension",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/github.com/prometheus/client_model/go:go_default_library",
        "//vendor/k8s.io/utils/clock:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["sampling-histogram_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/github.com/prometheus/client_model/go:go_default_library",
        "//vendor/k8s.io/utils/clock/testing:go_default_library",
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
