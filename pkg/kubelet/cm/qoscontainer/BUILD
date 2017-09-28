load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "qos_container_manager_linux.go",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "//pkg/api/v1/helper/qos:go_default_library",
            "//pkg/api/v1/resource:go_default_library",
            "//pkg/features:go_default_library",
            "//pkg/kubelet/cm:go_default_library",
            "//pkg/kubelet/cm/cgroupmanager:go_default_library",
            "//pkg/kubelet/cm/util:go_default_library",
            "//vendor/github.com/docker/go-units:go_default_library",
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/fs:go_default_library",
            "//vendor/k8s.io/api/core/v1:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
            "//vendor/k8s.io/apiserver/pkg/util/feature:go_default_library",
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
