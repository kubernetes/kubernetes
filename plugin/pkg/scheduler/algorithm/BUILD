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
        "listers.go",
        "scheduler_interface.go",
        "types.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/labels:go_default_library",
        "//plugin/pkg/scheduler/api:go_default_library",
        "//plugin/pkg/scheduler/schedulercache:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["scheduler_interface_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = ["//pkg/api/v1:go_default_library"],
)
