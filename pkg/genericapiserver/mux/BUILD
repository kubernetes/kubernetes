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
        "container.go",
        "doc.go",
        "pathrecorder.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/apiserver:go_default_library",
        "//pkg/runtime:go_default_library",
        "//vendor:github.com/emicklei/go-restful",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["container_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = ["//vendor:github.com/stretchr/testify/assert"],
)
