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
        "index.go",
        "metrics.go",
        "openapi.go",
        "profiling.go",
        "swagger.go",
        "swaggerui.go",
        "version.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/apiserver:go_default_library",
        "//pkg/apiserver/metrics:go_default_library",
        "//pkg/genericapiserver/mux:go_default_library",
        "//pkg/genericapiserver/openapi:go_default_library",
        "//pkg/genericapiserver/openapi/common:go_default_library",
        "//pkg/genericapiserver/routes/data/swagger:go_default_library",
        "//pkg/storage/etcd/metrics:go_default_library",
        "//pkg/version:go_default_library",
        "//vendor:github.com/elazarl/go-bindata-assetfs",
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/emicklei/go-restful/swagger",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/prometheus/client_golang/prometheus",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["swagger_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/genericapiserver/mux:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
    ],
)
