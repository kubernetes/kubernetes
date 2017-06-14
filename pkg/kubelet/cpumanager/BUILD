package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "cpu_manager_test.go",
        "manager_test.go",
        "state_mem_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = ["//pkg/kubelet/cpumanager/topo:go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = [
        "cpu_manager.go",
        "manager.go",
        "policy.go",
        "policy_noop.go",
        "policy_static.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/kubelet/apis/cri:go_default_library",
        "//pkg/kubelet/cpumanager/state:go_default_library",
        "//pkg/kubelet/cpumanager/topo:go_default_library",
        "//pkg/kubelet/cpuset:go_default_library",
        "//pkg/kubelet/qos:go_default_library",
        "//pkg/kubelet/status:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
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
        "//pkg/kubelet/cpumanager/state:all-srcs",
        "//pkg/kubelet/cpumanager/topo:all-srcs",
    ],
    tags = ["automanaged"],
)
