load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "fake_topology_manager.go",
        "policy.go",
        "policy_best_effort.go",
        "policy_none.go",
        "policy_restricted.go",
        "policy_single_numa_node.go",
        "topology_manager.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/cm/topologymanager",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubelet/cm/topologymanager/bitmask:go_default_library",
        "//pkg/kubelet/lifecycle:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
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
        "//pkg/kubelet/cm/topologymanager/bitmask:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "fake_topology_manager_test.go",
        "policy_best_effort_test.go",
        "policy_none_test.go",
        "policy_restricted_test.go",
        "policy_single_numa_node_test.go",
        "policy_test.go",
        "topology_manager_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/kubelet/cm/topologymanager/bitmask:go_default_library",
        "//pkg/kubelet/lifecycle:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
    ],
)
