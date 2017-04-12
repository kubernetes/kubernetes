package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "converter_test.go",
        "deep_copy_test.go",
        "helper_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//vendor/github.com/google/gofuzz:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/diff:go_default_library",
    ],
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
    deps = ["//vendor/k8s.io/apimachinery/third_party/forked/golang/reflect:go_default_library"],
)
