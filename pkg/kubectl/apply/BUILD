load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "element.go",
        "empty_element.go",
        "error.go",
        "list_element.go",
        "map_element.go",
        "primitive_element.go",
        "type_element.go",
        "visitor.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubectl/apply",
    visibility = ["//visibility:public"],
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
        "//pkg/kubectl/apply/parse:all-srcs",
        "//pkg/kubectl/apply/strategy:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
