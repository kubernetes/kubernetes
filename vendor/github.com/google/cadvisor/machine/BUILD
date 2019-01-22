load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "info.go",
        "machine.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/google/cadvisor/machine",
    importpath = "github.com/google/cadvisor/machine",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/docker/docker/pkg/parsers/operatingsystem:go_default_library",
        "//vendor/github.com/google/cadvisor/fs:go_default_library",
        "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
        "//vendor/github.com/google/cadvisor/utils:go_default_library",
        "//vendor/github.com/google/cadvisor/utils/cloudinfo:go_default_library",
        "//vendor/github.com/google/cadvisor/utils/sysfs:go_default_library",
        "//vendor/github.com/google/cadvisor/utils/sysinfo:go_default_library",
        "//vendor/golang.org/x/sys/unix:go_default_library",
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
