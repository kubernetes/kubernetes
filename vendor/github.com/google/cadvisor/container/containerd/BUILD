load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "client.go",
        "factory.go",
        "grpc.go",
        "handler.go",
        "plugin.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/google/cadvisor/container/containerd",
    importpath = "github.com/google/cadvisor/container/containerd",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/containerd/containerd/api/services/containers/v1:go_default_library",
        "//vendor/github.com/containerd/containerd/api/services/tasks/v1:go_default_library",
        "//vendor/github.com/containerd/containerd/api/services/version/v1:go_default_library",
        "//vendor/github.com/containerd/containerd/containers:go_default_library",
        "//vendor/github.com/containerd/containerd/errdefs:go_default_library",
        "//vendor/github.com/containerd/containerd/namespaces:go_default_library",
        "//vendor/github.com/containerd/containerd/pkg/dialer:go_default_library",
        "//vendor/github.com/gogo/protobuf/types:go_default_library",
        "//vendor/github.com/google/cadvisor/container:go_default_library",
        "//vendor/github.com/google/cadvisor/container/common:go_default_library",
        "//vendor/github.com/google/cadvisor/container/libcontainer:go_default_library",
        "//vendor/github.com/google/cadvisor/fs:go_default_library",
        "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
        "//vendor/github.com/google/cadvisor/watcher:go_default_library",
        "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/fs:go_default_library",
        "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
        "//vendor/github.com/opencontainers/runtime-spec/specs-go:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/google.golang.org/grpc:go_default_library",
        "//vendor/google.golang.org/grpc/backoff:go_default_library",
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
        "//vendor/github.com/google/cadvisor/container/containerd/install:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
