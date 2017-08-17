load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "compress.go",
        "compressor_cache.go",
        "compressor_pools.go",
        "compressors.go",
        "constants.go",
        "container.go",
        "cors_filter.go",
        "curly.go",
        "curly_route.go",
        "doc.go",
        "entity_accessors.go",
        "filter.go",
        "jsr311.go",
        "logger.go",
        "mime.go",
        "options_filter.go",
        "parameter.go",
        "path_expression.go",
        "request.go",
        "response.go",
        "route.go",
        "route_builder.go",
        "router.go",
        "service_error.go",
        "web_service.go",
        "web_service_container.go",
    ],
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/emicklei/go-restful/log:go_default_library"],
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
        "//vendor/github.com/emicklei/go-restful/log:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
