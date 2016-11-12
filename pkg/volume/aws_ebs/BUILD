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
        "aws_ebs.go",
        "aws_util.go",
        "doc.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "attacher_test.go",
        "aws_ebs_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
