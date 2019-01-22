# gazelle:exclude testdata

load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "go2make.go",
        "pkgwalk.go",
    ],
    importpath = "k8s.io/kubernetes/hack/make-rules/helpers/go2make",
    visibility = ["//visibility:private"],
    deps = ["//vendor/github.com/spf13/pflag:go_default_library"],
)

go_binary(
    name = "go2make",
    embed = [":go_default_library"],
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["pkgwalk_test.go"],
    data = glob(["testdata/**"]),
    embed = [":go_default_library"],
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
