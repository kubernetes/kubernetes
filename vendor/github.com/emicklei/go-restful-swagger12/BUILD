load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "api_declaration_list.go",
        "config.go",
        "model_builder.go",
        "model_list.go",
        "model_property_ext.go",
        "model_property_list.go",
        "ordered_route_map.go",
        "swagger.go",
        "swagger_builder.go",
        "swagger_webservice.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/emicklei/go-restful:go_default_library",
        "//vendor/github.com/emicklei/go-restful/log:go_default_library",
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
