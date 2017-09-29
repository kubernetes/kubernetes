load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bindata.go",
        "contact_info.go",
        "expander.go",
        "external_docs.go",
        "header.go",
        "info.go",
        "items.go",
        "license.go",
        "operation.go",
        "parameter.go",
        "path_item.go",
        "paths.go",
        "ref.go",
        "response.go",
        "responses.go",
        "schema.go",
        "security_scheme.go",
        "spec.go",
        "swagger.go",
        "tag.go",
        "xml_object.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/go-openapi/jsonpointer:go_default_library",
        "//vendor/github.com/go-openapi/jsonreference:go_default_library",
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
