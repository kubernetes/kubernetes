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
)
