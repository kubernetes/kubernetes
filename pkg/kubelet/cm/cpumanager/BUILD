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
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/api/v1/helper/qos:go_default_library",
        "//pkg/kubelet/apis/cri/v1alpha1/runtime:go_default_library",
        "//pkg/kubelet/cm/cpumanager/state:go_default_library",
        "//pkg/kubelet/cm/cpumanager/topology:go_default_library",
        "//pkg/kubelet/cm/cpuset:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/status:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
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
    ],
    library = ":go_default_library",
    deps = [
        "//pkg/kubelet/apis/cri/v1alpha1/runtime:go_default_library",
        "//pkg/kubelet/cm/cpumanager/state:go_default_library",
        "//pkg/kubelet/cm/cpumanager/topology:go_default_library",
        "//pkg/kubelet/cm/cpuset:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
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
