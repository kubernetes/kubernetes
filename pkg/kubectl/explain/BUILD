load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "explain.go",
        "field_builder.go",
        "fields.go",
        "formatter.go",
        "model.go",
        "recursive_fields.go",
        "schema_walker.go",
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

go_test(
    name = "go_default_test",
    srcs = [
        "fields_test.go",
        "formatter_test.go",
        "model_test.go",
        "recursive_fields_test.go",
        "schema_walker_test.go",
        "typename_test.go",
    ],
    data = ["test-swagger.json"],
    library = ":go_default_library",
    deps = [
        "//pkg/kubectl/cmd/util/openapi/testing:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
    ],
)
