package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "counter.go",
        "gauge.go",
        "histogram.go",
        "metric.go",
        "opts.go",
        "processstarttime.go",
        "registry.go",
        "summary.go",
        "version_parser.go",
        "wrappers.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/component-base/metrics",
    importpath = "k8s.io/component-base/metrics",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/version:go_default_library",
        "//vendor/github.com/blang/semver:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/github.com/prometheus/client_model/go:go_default_library",
        "//vendor/github.com/prometheus/procfs:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "counter_test.go",
        "gauge_test.go",
        "histogram_test.go",
        "registry_test.go",
        "summary_test.go",
        "version_parser_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/version:go_default_library",
        "//vendor/github.com/blang/semver:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/github.com/prometheus/common/expfmt:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
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
        "//staging/src/k8s.io/component-base/metrics/legacyregistry:all-srcs",
    ],
    tags = ["automanaged"],
)
