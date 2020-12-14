load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["mock_client.go"],
    importpath = "k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker/testing",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubelet/dockershim/libdocker:go_default_library",
        "//vendor/github.com/docker/docker/api/types:go_default_library",
        "//vendor/github.com/docker/docker/api/types/container:go_default_library",
        "//vendor/github.com/docker/docker/api/types/image:go_default_library",
        "//vendor/github.com/golang/mock/gomock:go_default_library",
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
    visibility = ["//visibility:public"],
)
