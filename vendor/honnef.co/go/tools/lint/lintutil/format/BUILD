load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["format.go"],
    importmap = "k8s.io/kubernetes/vendor/honnef.co/go/tools/lint/lintutil/format",
    importpath = "honnef.co/go/tools/lint/lintutil/format",
    visibility = ["//visibility:public"],
    deps = ["//vendor/honnef.co/go/tools/lint:go_default_library"],
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
