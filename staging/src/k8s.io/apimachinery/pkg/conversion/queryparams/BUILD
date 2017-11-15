load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "convert.go",
        "doc.go",
    ],
    importpath = "k8s.io/apimachinery/pkg/conversion/queryparams",
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_xtest",
    srcs = ["convert_test.go"],
    importpath = "k8s.io/apimachinery/pkg/conversion/queryparams_test",
    deps = [
        ":go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/runtime/schema:go_default_library",
    ],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
