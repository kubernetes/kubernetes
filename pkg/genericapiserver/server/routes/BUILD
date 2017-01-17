package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
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
        "//pkg/genericapiserver/endpoints/handlers/responsewriters:go_default_library",
        "//pkg/genericapiserver/server/mux:go_default_library",
        "//pkg/genericapiserver/server/openapi:go_default_library",
        "//pkg/genericapiserver/server/routes/data/swagger:go_default_library",
        "//vendor:github.com/elazarl/go-bindata-assetfs",
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/emicklei/go-restful/swagger",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/prometheus/client_golang/prometheus",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/openapi",
        "//vendor:k8s.io/apimachinery/pkg/version",
        "//vendor:k8s.io/apiserver/pkg/endpoints/metrics",
        "//vendor:k8s.io/apiserver/pkg/storage/etcd/metrics",
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
        "//pkg/genericapiserver/server/routes/data/swagger:all-srcs",
    ],
    tags = ["automanaged"],
)
