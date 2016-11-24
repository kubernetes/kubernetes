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
        "doc.go",
        "envvars.go",
    ],
    tags = ["automanaged"],
    deps = ["//pkg/api/v1:go_default_library"],
)

go_test(
    name = "go_default_xtest",
    srcs = ["envvars_test.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/kubelet/envvars:go_default_library",
    ],
)
