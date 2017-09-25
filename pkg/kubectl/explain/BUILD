load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "explain.go",
        "field_lookup.go",
        "fields_printer.go",
        "fields_printer_builder.go",
        "formatter.go",
        "model_printer.go",
        "recursive_fields_printer.go",
        "typename.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubectl/cmd/util/openapi:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/meta:go_default_library",
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
