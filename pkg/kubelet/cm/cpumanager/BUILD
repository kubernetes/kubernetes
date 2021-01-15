load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "cpu_assignment.go",
        "cpu_manager.go",
        "fake_cpu_manager.go",
        "policy.go",
        "policy_none.go",
        "policy_static.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/cm/cpumanager",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/apis/core/v1/helper/qos:go_default_library",
        "//pkg/kubelet/cm/containermap:go_default_library",
        "//pkg/kubelet/cm/cpumanager/state:go_default_library",
        "//pkg/kubelet/cm/cpumanager/topology:go_default_library",
        "//pkg/kubelet/cm/cpuset:go_default_library",
        "//pkg/kubelet/cm/topologymanager:go_default_library",
        "//pkg/kubelet/cm/topologymanager/bitmask:go_default_library",
        "//pkg/kubelet/config:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/status:go_default_library",
        "//pkg/kubelet/util/format:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/cri-api/pkg/apis/runtime/v1alpha2:go_default_library",
        "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "cpu_assignment_test.go",
        "cpu_manager_test.go",
        "policy_none_test.go",
        "policy_static_test.go",
        "policy_test.go",
        "topology_hints_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/kubelet/cm/containermap:go_default_library",
        "//pkg/kubelet/cm/cpumanager/state:go_default_library",
        "//pkg/kubelet/cm/cpumanager/topology:go_default_library",
        "//pkg/kubelet/cm/cpuset:go_default_library",
        "//pkg/kubelet/cm/topologymanager:go_default_library",
        "//pkg/kubelet/cm/topologymanager/bitmask:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/cri-api/pkg/apis/runtime/v1alpha2:go_default_library",
        "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
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
        "//pkg/kubelet/cm/cpumanager/state:all-srcs",
        "//pkg/kubelet/cm/cpumanager/topology:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
