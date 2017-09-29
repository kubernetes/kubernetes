load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bytestream.go",
        "client_auth_info.go",
        "client_operation.go",
        "client_request.go",
        "client_response.go",
        "constants.go",
        "discard.go",
        "headers.go",
        "interfaces.go",
        "json.go",
        "request.go",
        "statuses.go",
        "text.go",
        "values.go",
        "xml.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/go-openapi/errors:go_default_library",
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
