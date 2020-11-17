load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "debug.go",
        "doc.go",
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
        "type.go",
        "validator.go",
        "values.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kube-openapi/pkg/validation/validate",
    importpath = "k8s.io/kube-openapi/pkg/validation/validate",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/go-openapi/swag:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/validation/errors:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/validation/spec:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/validation/strfmt:go_default_library",
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
