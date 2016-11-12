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
        "cleanup.go",
        "exec_util.go",
        "federation_util.go",
        "framework.go",
        "gobindata_util.go",
        "kubelet_stats.go",
        "log_size_monitoring.go",
        "metrics_util.go",
        "networking_utils.go",
        "nodes_util.go",
        "perf_util.go",
        "pods.go",
        "resource_usage_gatherer.go",
        "test_context.go",
        "util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/onsi/ginkgo",
        "//vendor:github.com/onsi/ginkgo/config",
        "//vendor:github.com/onsi/gomega",
        "//vendor:github.com/onsi/gomega/types",
        "//vendor:gopkg.in/yaml.v2",
    ],
)
