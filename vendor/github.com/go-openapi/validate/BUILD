load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "debug.go",
        "default_validator.go",
        "doc.go",
        "example_validator.go",
        "formats.go",
        "helpers.go",
        "object_validator.go",
        "options.go",
        "result.go",
        "rexp.go",
        "schema.go",
        "schema_messages.go",
        "schema_option.go",
        "schema_props.go",
        "slice_validator.go",
        "spec.go",
        "spec_messages.go",
        "type.go",
        "validator.go",
        "values.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/go-openapi/validate",
    importpath = "github.com/go-openapi/validate",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/go-openapi/analysis:go_default_library",
        "//vendor/github.com/go-openapi/errors:go_default_library",
        "//vendor/github.com/go-openapi/jsonpointer:go_default_library",
        "//vendor/github.com/go-openapi/loads:go_default_library",
        "//vendor/github.com/go-openapi/runtime:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
