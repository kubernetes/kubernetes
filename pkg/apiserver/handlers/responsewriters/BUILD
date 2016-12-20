package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "writers.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/rest:go_default_library",
        "//pkg/apiserver/handlers/errors:go_default_library",
        "//pkg/apiserver/handlers/negotiation:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/util/flushwriter:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/util/wsstream:go_default_library",
    ],
)
