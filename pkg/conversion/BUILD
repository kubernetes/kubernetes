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
        "cloner.go",
        "converter.go",
        "deep_equal.go",
        "doc.go",
        "helper.go",
    ],
    tags = ["automanaged"],
    deps = ["//third_party/forked/golang/reflect:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "converter_test.go",
        "deep_copy_test.go",
        "helper_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/util/diff:go_default_library",
        "//vendor:github.com/google/gofuzz",
        "//vendor:github.com/spf13/pflag",
    ],
)
