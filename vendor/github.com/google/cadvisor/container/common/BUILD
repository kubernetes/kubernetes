load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "container_hints.go",
        "fsHandler.go",
        "helpers.go",
        "inotify_watcher.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/google/cadvisor/container/common",
    importpath = "github.com/google/cadvisor/container/common",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/google/cadvisor/container:go_default_library",
        "//vendor/github.com/google/cadvisor/fs:go_default_library",
        "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
        "//vendor/github.com/google/cadvisor/utils:go_default_library",
        "//vendor/github.com/karrick/godirwalk:go_default_library",
        "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
        "//vendor/k8s.io/utils/inotify:go_default_library",
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
