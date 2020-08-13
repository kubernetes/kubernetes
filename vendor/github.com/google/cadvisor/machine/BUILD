load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "info.go",
        "machine.go",
        "operatingsystem_unix.go",
        "operatingsystem_windows.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/google/cadvisor/machine",
    importpath = "github.com/google/cadvisor/machine",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/google/cadvisor/fs:go_default_library",
        "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
        "//vendor/github.com/google/cadvisor/nvm:go_default_library",
        "//vendor/github.com/google/cadvisor/utils:go_default_library",
        "//vendor/github.com/google/cadvisor/utils/cloudinfo:go_default_library",
        "//vendor/github.com/google/cadvisor/utils/sysfs:go_default_library",
        "//vendor/github.com/google/cadvisor/utils/sysinfo:go_default_library",
        "//vendor/golang.org/x/sys/unix:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/golang.org/x/sys/windows/registry:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
