package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "container_manager.go",
        "container_manager_unsupported.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "container_manager_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "container_manager_windows.go",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//pkg/kubelet/dockershim/libdocker:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "//pkg/kubelet/cm:go_default_library",
            "//pkg/kubelet/qos:go_default_library",
            "//pkg/util/version:go_default_library",
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/fs:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
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
)
