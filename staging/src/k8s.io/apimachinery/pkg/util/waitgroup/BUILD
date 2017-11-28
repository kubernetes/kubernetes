load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "waitgroup.go",
    ],
    importpath = "k8s.io/apimachinery/pkg/util/waitgroup",
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["waitgroup_test.go"],
    importpath = "k8s.io/apimachinery/pkg/util/waitgroup",
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
