package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["status_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/runtime/schema:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "errors.go",
        "status.go",
        "writers.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/rest:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/auth/authorizer:go_default_library",
        "//pkg/genericapiserver/api/handlers/negotiation:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/storage:go_default_library",
        "//pkg/util/flushwriter:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/util/wsstream:go_default_library",
    ],
)
