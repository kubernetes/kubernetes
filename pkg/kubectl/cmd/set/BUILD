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
        "helper.go",
        "set.go",
        "set_image.go",
        "set_resources.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = ["set_image_test.go"],
    data = [
        "//examples:config",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
