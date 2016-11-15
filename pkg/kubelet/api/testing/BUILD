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
        "fake_image_service.go",
        "fake_runtime_service.go",
        "utils.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/kubelet/api/v1alpha1/runtime:go_default_library",
        "//pkg/kubelet/util/sliceutils:go_default_library",
    ],
)
