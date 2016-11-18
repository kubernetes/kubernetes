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
        "helpers.go",
        "validate.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/util:go_default_library",
        "//pkg/util/config:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["validate_test.go"],
    data = [
        "testdata/profiles",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
    ],
)
