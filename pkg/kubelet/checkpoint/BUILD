load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "checkpoint.go",
        "metadata.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/checkpoint",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubelet/pod:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/github.com/checkpoint-restore/checkpointctl/lib:go_default_library",
        "//vendor/github.com/emicklei/go-restful:go_default_library",
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
