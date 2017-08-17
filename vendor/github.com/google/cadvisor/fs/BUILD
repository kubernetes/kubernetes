load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "types.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "fs.go",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "//vendor/github.com/docker/docker/pkg/mount:go_default_library",
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/github.com/google/cadvisor/devicemapper:go_default_library",
            "//vendor/github.com/google/cadvisor/utils/docker:go_default_library",
            "//vendor/github.com/mistifyio/go-zfs:go_default_library",
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
