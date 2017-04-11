package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "api_server_metrics.go",
        "controller_manager_metrics.go",
        "generic_metrics.go",
        "kubelet_metrics.go",
        "metrics_grabber.go",
        "scheduler_metrics.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/master/ports:go_default_library",
        "//pkg/util/system:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/prometheus/common/expfmt",
        "//vendor:github.com/prometheus/common/model",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
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
