load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "cgroup_manager_unsupported.go",
        "container_manager.go",
        "container_manager_stub.go",
        "container_manager_unsupported.go",
        "device_plugin_handler.go",
        "device_plugin_handler_stub.go",
        "fake_internal_container_lifecycle.go",
        "helpers_unsupported.go",
        "internal_container_lifecycle.go",
        "pod_container_manager_stub.go",
        "pod_container_manager_unsupported.go",
        "types.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "cgroup_manager_linux.go",
            "container_manager_linux.go",
            "helpers_linux.go",
            "node_container_manager.go",
            "pod_container_manager_linux.go",
            "qos_container_manager_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "container_manager_windows.go",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/features:go_default_library",
        "//pkg/kubelet/apis/cri:go_default_library",
        "//pkg/kubelet/apis/deviceplugin/v1alpha1:go_default_library",
        "//pkg/kubelet/apis/kubeletconfig:go_default_library",
        "//pkg/kubelet/cadvisor:go_default_library",
        "//pkg/kubelet/cm/cpumanager:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/deviceplugin:go_default_library",
        "//pkg/kubelet/eviction/api:go_default_library",
        "//pkg/kubelet/status:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//vendor/k8s.io/client-go/tools/record:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "//pkg/api:go_default_library",
            "//pkg/api/v1/helper:go_default_library",
            "//pkg/api/v1/helper/qos:go_default_library",
            "//pkg/api/v1/resource:go_default_library",
            "//pkg/kubelet/cm/util:go_default_library",
            "//pkg/kubelet/events:go_default_library",
            "//pkg/kubelet/metrics:go_default_library",
            "//pkg/kubelet/qos:go_default_library",
            "//pkg/util/file:go_default_library",
            "//pkg/util/oom:go_default_library",
            "//pkg/util/procfs:go_default_library",
            "//pkg/util/sysctl:go_default_library",
            "//pkg/util/version:go_default_library",
            "//vendor/github.com/docker/go-units:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/fs:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/systemd:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = [
        "container_manager_unsupported_test.go",
        "device_plugin_handler_test.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "cgroup_manager_linux_test.go",
            "cgroup_manager_test.go",
            "container_manager_linux_test.go",
            "helpers_linux_test.go",
            "node_container_manager_test.go",
        ],
        "//conditions:default": [],
    }),
    library = ":go_default_library",
    deps = [
        "//pkg/kubelet/apis/deviceplugin/v1alpha1:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "//pkg/kubelet/apis/kubeletconfig:go_default_library",
            "//pkg/kubelet/eviction/api:go_default_library",
            "//vendor/github.com/stretchr/testify/require:go_default_library",
        ],
        "//conditions:default": [],
    }),
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
        "//pkg/kubelet/cm/cpumanager:all-srcs",
        "//pkg/kubelet/cm/cpuset:all-srcs",
        "//pkg/kubelet/cm/util:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
