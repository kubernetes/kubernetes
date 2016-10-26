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
        "app-interface.go",
        "doc.go",
        "fake-app-interface.go",
        "imagestore.go",
        "pod-level-interface.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/kubelet/api:go_default_library",
        "//pkg/kubelet/api/v1alpha1/runtime:go_default_library",
        "//pkg/kubelet/util/ioutils:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["imagestore_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = ["//pkg/kubelet/api/v1alpha1/runtime:go_default_library"],
)
