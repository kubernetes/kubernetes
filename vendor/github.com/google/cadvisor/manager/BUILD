load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "container.go",
        "manager.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/google/cadvisor/manager",
    importpath = "github.com/google/cadvisor/manager",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/docker/go-units:go_default_library",
        "//vendor/github.com/google/cadvisor/accelerators:go_default_library",
        "//vendor/github.com/google/cadvisor/cache/memory:go_default_library",
        "//vendor/github.com/google/cadvisor/collector:go_default_library",
        "//vendor/github.com/google/cadvisor/container:go_default_library",
        "//vendor/github.com/google/cadvisor/container/docker:go_default_library",
        "//vendor/github.com/google/cadvisor/container/raw:go_default_library",
        "//vendor/github.com/google/cadvisor/events:go_default_library",
        "//vendor/github.com/google/cadvisor/fs:go_default_library",
        "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
        "//vendor/github.com/google/cadvisor/info/v2:go_default_library",
        "//vendor/github.com/google/cadvisor/machine:go_default_library",
        "//vendor/github.com/google/cadvisor/nvm:go_default_library",
        "//vendor/github.com/google/cadvisor/perf:go_default_library",
        "//vendor/github.com/google/cadvisor/stats:go_default_library",
        "//vendor/github.com/google/cadvisor/summary:go_default_library",
        "//vendor/github.com/google/cadvisor/utils/cpuload:go_default_library",
        "//vendor/github.com/google/cadvisor/utils/oomparser:go_default_library",
        "//vendor/github.com/google/cadvisor/utils/sysfs:go_default_library",
        "//vendor/github.com/google/cadvisor/version:go_default_library",
        "//vendor/github.com/google/cadvisor/watcher:go_default_library",
        "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
        "//vendor/k8s.io/utils/clock:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
