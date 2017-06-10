package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
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
        "//pkg/kubelet/apis/cri:go_default_library",
        "//pkg/kubelet/apis/cri/v1alpha1/runtime:go_default_library",
        "//pkg/kubelet/util/ioutils:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["imagestore_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = ["//pkg/kubelet/apis/cri/v1alpha1/runtime:go_default_library"],
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
