load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "path.go",
        "timed_store.go",
        "utils.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/google/cadvisor/utils",
    importpath = "github.com/google/cadvisor/utils",
    visibility = ["//visibility:public"],
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
        "//vendor/github.com/google/cadvisor/utils/cloudinfo:all-srcs",
        "//vendor/github.com/google/cadvisor/utils/cpuload:all-srcs",
        "//vendor/github.com/google/cadvisor/utils/oomparser:all-srcs",
        "//vendor/github.com/google/cadvisor/utils/sysfs:all-srcs",
        "//vendor/github.com/google/cadvisor/utils/sysinfo:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
