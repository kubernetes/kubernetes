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
        "errors.go",
        "firsthit_restmapper.go",
        "help.go",
        "interfaces.go",
        "meta.go",
        "multirestmapper.go",
        "priority.go",
        "restmapper.go",
        "unstructured.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "multirestmapper_test.go",
        "priority_test.go",
        "restmapper_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)

go_test(
    name = "go_default_xtest",
    srcs = [
        "help_test.go",
        "meta_test.go",
        "scheme_test.go",
    ],
    tags = ["automanaged"],
    deps = [],
)
