load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["fuzzer.go"],
    importpath = "k8s.io/kubernetes/cmd/kubelet/pkg/apis/config/fuzzer",
    visibility = ["//visibility:public"],
    deps = [
        "//cmd/kubelet/pkg/apis/config:go_default_library",
        "//cmd/kubelet/pkg/apis/config/v1beta1:go_default_library",
        "//cmd/kubelet/pkg/qos:go_default_library",
        "//cmd/kubelet/pkg/types:go_default_library",
        "//pkg/master/ports:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//staging/src/k8s.io/kubelet/config/v1beta1:go_default_library",
        "//vendor/github.com/google/gofuzz:go_default_library",
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
