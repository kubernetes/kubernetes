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
        "evaluator.go",
        "registry.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/admission:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/controller/informers:go_default_library",
        "//pkg/labels:go_default_library",
        "//pkg/quota:go_default_library",
        "//pkg/runtime:go_default_library",
    ],
)
