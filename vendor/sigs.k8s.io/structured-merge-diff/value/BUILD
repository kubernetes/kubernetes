load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "fields.go",
        "list.go",
        "listinterface.go",
        "map.go",
        "mapinterface.go",
        "reflect.go",
        "scalar.go",
        "value.go",
        "valueinterface.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/structured-merge-diff/value",
    importpath = "sigs.k8s.io/structured-merge-diff/value",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/json-iterator/go:go_default_library"],
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
