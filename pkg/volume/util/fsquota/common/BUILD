load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "quota_linux_common.go",
        "quota_linux_common_impl.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/volume/util/fsquota/common",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/k8s.io/klog:go_default_library",
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
