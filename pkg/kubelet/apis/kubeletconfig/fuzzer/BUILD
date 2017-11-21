load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["fuzzer.go"],
    importpath = "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/fuzzer",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubelet/apis/kubeletconfig:go_default_library",
        "//pkg/kubelet/apis/kubeletconfig/v1alpha1:go_default_library",
        "//pkg/kubelet/qos:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//pkg/master/ports:go_default_library",
        "//vendor/github.com/google/gofuzz:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
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
