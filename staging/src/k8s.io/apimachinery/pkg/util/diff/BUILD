load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["diff.go"],
    importpath = "k8s.io/apimachinery/pkg/util/diff",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/util/validation/field:go_default_library",
        "//vendor/github.com/davecgh/go-spew/spew:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["diff_test.go"],
    importpath = "k8s.io/apimachinery/pkg/util/diff",
    library = ":go_default_library",
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
