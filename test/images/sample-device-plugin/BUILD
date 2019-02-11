package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["sampledeviceplugin.go"],
    importpath = "k8s.io/kubernetes/test/images/sample-device-plugin",
    visibility = ["//visibility:private"],
    deps = [
        "//pkg/kubelet/apis/deviceplugin/v1beta1:go_default_library",
        "//pkg/kubelet/cm/devicemanager:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
    ],
)

go_binary(
    name = "sample-device-plugin",
    embed = [":go_default_library"],
    visibility = ["//visibility:public"],
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
