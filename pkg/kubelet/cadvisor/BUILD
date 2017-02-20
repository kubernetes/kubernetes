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
        "cadvisor_linux.go",
        "doc.go",
        "types.go",
        "util.go",
    ],
    tags = ["automanaged"],
    deps = ["//pkg/kubelet/types:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = ["cadvisor_linux_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = ["//pkg/kubelet/types:go_default_library"],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//pkg/kubelet/cadvisor/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
