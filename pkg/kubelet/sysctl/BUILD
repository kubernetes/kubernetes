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
        "namespace.go",
        "runtime.go",
        "whitelist.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/validation:go_default_library",
        "//pkg/apis/extensions/validation:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/dockertools:go_default_library",
        "//pkg/kubelet/lifecycle:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "namespace_test.go",
        "whitelist_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = ["//pkg/api:go_default_library"],
)
