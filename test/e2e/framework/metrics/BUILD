package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "api_server_metrics.go",
        "cluster_autoscaler_metrics.go",
        "controller_manager_metrics.go",
        "generic_metrics.go",
        "kubelet_metrics.go",
        "metrics_grabber.go",
        "scheduler_metrics.go",
    ],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/master/ports:go_default_library",
        "//pkg/util/system:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/prometheus/common/expfmt:go_default_library",
        "//vendor/github.com/prometheus/common/model:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
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
