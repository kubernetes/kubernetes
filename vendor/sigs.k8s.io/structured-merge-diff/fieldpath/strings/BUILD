load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "reader.go",
        "stream.go",
        "table.go",
        "v0.go",
        "v1.go",
        "v2.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/structured-merge-diff/fieldpath/strings",
    importpath = "sigs.k8s.io/structured-merge-diff/fieldpath/strings",
    visibility = ["//visibility:public"],
    deps = ["//vendor/sigs.k8s.io/structured-merge-diff/value:go_default_library"],
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
