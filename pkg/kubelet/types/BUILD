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
        "constants.go",
        "doc.go",
        "labels.go",
        "pod_update.go",
        "types.go",
    ],
    tags = ["automanaged"],
    deps = ["//pkg/api:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "pod_update_test.go",
        "types_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//vendor:github.com/stretchr/testify/require",
    ],
)
