load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "log_metrics.go",
        "resource_metrics.go",
        "volume_stats.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/metrics/collectors",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubelet/metrics:go_default_library",
        "//pkg/kubelet/server/stats:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/component-base/metrics:go_default_library",
        "//staging/src/k8s.io/kubelet/pkg/apis/stats/v1alpha1:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "log_metrics_test.go",
        "resource_metrics_test.go",
        "volume_stats_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/kubelet/server/stats/testing:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/component-base/metrics/testutil:go_default_library",
        "//staging/src/k8s.io/kubelet/pkg/apis/stats/v1alpha1:go_default_library",
        "//vendor/github.com/stretchr/testify/mock:go_default_library",
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
