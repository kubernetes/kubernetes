package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "fs.go",
        "types.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/github.com/docker/docker/pkg/mount:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/google/cadvisor/devicemapper:go_default_library",
        "//vendor/github.com/google/cadvisor/utils/docker:go_default_library",
        "//vendor/github.com/mistifyio/go-zfs:go_default_library",
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
)
