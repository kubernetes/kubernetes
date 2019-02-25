load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "client.go",
        "factory.go",
        "handler.go",
        "mesos_agent.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/google/cadvisor/container/mesos",
    importpath = "github.com/google/cadvisor/container/mesos",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/Rican7/retry:go_default_library",
        "//vendor/github.com/Rican7/retry/strategy:go_default_library",
        "//vendor/github.com/google/cadvisor/container:go_default_library",
        "//vendor/github.com/google/cadvisor/container/common:go_default_library",
        "//vendor/github.com/google/cadvisor/container/libcontainer:go_default_library",
        "//vendor/github.com/google/cadvisor/fs:go_default_library",
        "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
        "//vendor/github.com/google/cadvisor/manager/watcher:go_default_library",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib:go_default_library",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/agent:go_default_library",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/agent/calls:go_default_library",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/client:go_default_library",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/encoding/codecs:go_default_library",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/httpcli:go_default_library",
        "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/fs:go_default_library",
        "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
