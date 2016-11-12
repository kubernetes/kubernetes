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
        "attacher.go",
        "cinder.go",
        "cinder_util.go",
        "doc.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "attacher_test.go",
        "cinder_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
