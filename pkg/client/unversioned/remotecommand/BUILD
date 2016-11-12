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
        "errorstream.go",
        "remotecommand.go",
        "v1.go",
        "v2.go",
        "v3.go",
        "v4.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "remotecommand_test.go",
        "v2_test.go",
        "v4_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
