package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "container_manager.go",
        "container_manager_linux.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/kubelet/cm:go_default_library",
        "//pkg/kubelet/dockertools:go_default_library",
        "//pkg/kubelet/qos:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//vendor:github.com/blang/semver",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/opencontainers/runc/libcontainer/cgroups/fs",
        "//vendor:github.com/opencontainers/runc/libcontainer/configs",
    ],
)
