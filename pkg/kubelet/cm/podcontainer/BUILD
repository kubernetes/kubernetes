load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "pod_container_manager_stub.go",
        "pod_container_manager_unsupported.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "pod_container_manager_linux.go",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubelet/cm:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "//pkg/api/v1/helper/qos:go_default_library",
            "//pkg/kubelet/cm/util:go_default_library",
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
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
