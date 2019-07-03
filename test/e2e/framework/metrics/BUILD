package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "api.go",
        "api_server_metrics.go",
        "cluster_autoscaler_metrics.go",
        "controller_manager_metrics.go",
        "e2e_metrics.go",
        "etcd.go",
        "generic_metrics.go",
        "interesting_metrics.go",
        "kubelet_metrics.go",
        "latencies.go",
        "metrics_grabber.go",
        "pod.go",
        "scheduler_metrics.go",
        "scheduling.go",
    ],
    importpath = "k8s.io/kubernetes/test/e2e/framework/metrics",
    deps = [
        "//pkg/apis/core:go_default_library",
        "//pkg/kubelet/dockershim/metrics:go_default_library",
        "//pkg/kubelet/metrics:go_default_library",
        "//pkg/master/ports:go_default_library",
        "//pkg/scheduler/metrics:go_default_library",
        "//pkg/util/system:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//test/e2e/framework/log:go_default_library",
        "//test/e2e/framework/ssh:go_default_library",
        "//test/e2e/perftype:go_default_library",
        "//vendor/github.com/onsi/gomega:go_default_library",
        "//vendor/github.com/prometheus/common/expfmt:go_default_library",
        "//vendor/github.com/prometheus/common/model:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
