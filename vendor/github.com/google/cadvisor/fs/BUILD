load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "fs.go",
        "types.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/google/cadvisor/fs",
    importpath = "github.com/google/cadvisor/fs",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/github.com/google/cadvisor/devicemapper:go_default_library",
            "//vendor/github.com/google/cadvisor/utils:go_default_library",
            "//vendor/github.com/mistifyio/go-zfs:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
            "//vendor/k8s.io/utils/mount:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/google/cadvisor/devicemapper:go_default_library",
            "//vendor/github.com/google/cadvisor/utils:go_default_library",
            "//vendor/github.com/mistifyio/go-zfs:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
            "//vendor/k8s.io/utils/mount:go_default_library",
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
