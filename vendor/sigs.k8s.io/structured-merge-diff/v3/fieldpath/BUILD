load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "element.go",
        "fromvalue.go",
        "managers.go",
        "path.go",
        "pathelementmap.go",
        "serialize.go",
        "serialize-pe.go",
        "set.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/structured-merge-diff/v3/fieldpath",
    importpath = "sigs.k8s.io/structured-merge-diff/v3/fieldpath",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/json-iterator/go:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/v3/value:go_default_library",
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
