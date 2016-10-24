package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "openapi.go",
        "util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/genericapiserver/mux:go_default_library",
        "//pkg/genericapiserver/openapi/common:go_default_library",
        "//pkg/util:go_default_library",
        "//pkg/util/json:go_default_library",
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/go-openapi/spec",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["openapi_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/genericapiserver/openapi/common:go_default_library",
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/go-openapi/spec",
        "//vendor:github.com/stretchr/testify/assert",
    ],
)
