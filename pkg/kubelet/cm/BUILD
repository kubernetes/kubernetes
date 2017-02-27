package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "cgroup_manager_linux.go",
        "container_manager.go",
        "container_manager_linux.go",
        "container_manager_stub.go",
        "helpers_linux.go",
        "node_container_manager.go",
        "pod_container_manager_linux.go",
        "pod_container_manager_stub.go",
        "qos_container_manager_linux.go",
        "types.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/componentconfig:go_default_library",
        "//pkg/kubelet/cadvisor:go_default_library",
        "//pkg/kubelet/cm/util:go_default_library",
        "//pkg/kubelet/events:go_default_library",
        "//pkg/kubelet/eviction/api:go_default_library",
        "//pkg/kubelet/metrics:go_default_library",
        "//pkg/kubelet/qos:go_default_library",
        "//pkg/util:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/util/oom:go_default_library",
        "//pkg/util/procfs:go_default_library",
        "//pkg/util/sysctl:go_default_library",
        "//pkg/util/version:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/opencontainers/runc/libcontainer/cgroups",
        "//vendor:github.com/opencontainers/runc/libcontainer/cgroups/fs",
        "//vendor:github.com/opencontainers/runc/libcontainer/cgroups/systemd",
        "//vendor:github.com/opencontainers/runc/libcontainer/configs",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/client-go/tools/record",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "cgroup_manager_linux_test.go",
        "cgroup_manager_test.go",
        "container_manager_linux_test.go",
        "helpers_linux_test.go",
        "node_container_manager_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/componentconfig:go_default_library",
        "//pkg/kubelet/eviction/api:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:github.com/stretchr/testify/require",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
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
        "//pkg/kubelet/cm/util:all-srcs",
    ],
    tags = ["automanaged"],
)
