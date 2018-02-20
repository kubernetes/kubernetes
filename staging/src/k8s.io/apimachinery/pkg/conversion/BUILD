package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "converter_test.go",
        "helper_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//vendor/github.com/google/gofuzz:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/diff:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "converter.go",
        "deep_equal.go",
        "doc.go",
        "helper.go",
    ],
    importpath = "k8s.io/apimachinery/pkg/conversion",
    deps = ["//vendor/k8s.io/apimachinery/third_party/forked/golang/reflect:go_default_library"],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//staging/src/k8s.io/apimachinery/pkg/conversion/queryparams:all-srcs",
    ],
    tags = ["automanaged"],
)
