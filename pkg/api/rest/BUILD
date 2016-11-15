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
        "create.go",
        "delete.go",
        "doc.go",
        "export.go",
        "rest.go",
        "types.go",
        "update.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/meta:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/api/validation:go_default_library",
        "//pkg/api/validation/path:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/validation/field:go_default_library",
        "//pkg/watch:go_default_library",
    ],
)
