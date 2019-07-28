load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "analyzer.go",
        "debug.go",
        "doc.go",
        "fixer.go",
        "flatten.go",
        "mixin.go",
        "schema.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/go-openapi/analysis",
    importpath = "github.com/go-openapi/analysis",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/go-openapi/analysis/internal:go_default_library",
        "//vendor/github.com/go-openapi/jsonpointer:go_default_library",
        "//vendor/github.com/go-openapi/spec:go_default_library",
        "//vendor/github.com/go-openapi/strfmt:go_default_library",
        "//vendor/github.com/go-openapi/swag:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/go-openapi/analysis/internal:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
