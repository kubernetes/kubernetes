package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
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
        "//pkg/util:go_default_library",
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/go-openapi/spec",
        "//vendor:k8s.io/apimachinery/pkg/openapi",
        "//vendor:k8s.io/apimachinery/pkg/util/json",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["openapi_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/go-openapi/spec",
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:k8s.io/apimachinery/pkg/openapi",
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
        "//pkg/genericapiserver/openapi/common:all-srcs",
    ],
    tags = ["automanaged"],
)
