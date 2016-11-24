package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
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
        "//pkg/api/v1:go_default_library",
        "//pkg/client/clientset_generated/release_1_5:go_default_library",
        "//pkg/fields:go_default_library",
        "//pkg/master/ports:go_default_library",
        "//pkg/util/system:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/prometheus/common/expfmt",
        "//vendor:github.com/prometheus/common/model",
    ],
)
