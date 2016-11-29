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
        "cgroup_manager_linux.go",
        "container_manager.go",
        "container_manager_linux.go",
        "container_manager_stub.go",
        "helpers_linux.go",
        "pod_container_manager_linux.go",
        "pod_container_manager_stub.go",
        "types.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/resource:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/kubelet/cadvisor:go_default_library",
        "//pkg/kubelet/cm/util:go_default_library",
        "//pkg/kubelet/qos:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util:go_default_library",
        "//pkg/util/errors:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/util/oom:go_default_library",
        "//pkg/util/procfs:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/sysctl:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//vendor:github.com/blang/semver",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/opencontainers/runc/libcontainer/cgroups",
        "//vendor:github.com/opencontainers/runc/libcontainer/cgroups/fs",
        "//vendor:github.com/opencontainers/runc/libcontainer/cgroups/systemd",
        "//vendor:github.com/opencontainers/runc/libcontainer/configs",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "cgroup_manager_linux_test.go",
        "container_manager_linux_test.go",
        "helpers_linux_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/resource:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:github.com/stretchr/testify/require",
    ],
)
