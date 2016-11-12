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
        "benchmark_util.go",
        "container.go",
        "doc.go",
        "image_list.go",
        "resource_collector.go",
        "simple_mount.go",
        "util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/onsi/ginkgo",
        "//vendor:github.com/onsi/gomega",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "apparmor_test.go",
        "cgroup_manager_test.go",
        "container_manager_test.go",
        "density_test.go",
        "disk_eviction_test.go",
        "dynamic_kubelet_configuration_test.go",
        "e2e_node_suite_test.go",
        "image_id_test.go",
        "kubelet_test.go",
        "lifecycle_hook_test.go",
        "log_path_test.go",
        "memory_eviction_test.go",
        "mirror_pod_test.go",
        "resource_usage_test.go",
        "restart_test.go",
        "runtime_conformance_test.go",
        "summary_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/onsi/ginkgo",
        "//vendor:github.com/onsi/ginkgo/config",
        "//vendor:github.com/onsi/ginkgo/reporters",
        "//vendor:github.com/onsi/gomega",
        "//vendor:github.com/onsi/gomega/gstruct",
        "//vendor:github.com/onsi/gomega/types",
    ],
)
