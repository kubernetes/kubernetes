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
        "logs.go",
        "metrics.go",
        "ui.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/apiserver/metrics:go_default_library",
        "//pkg/genericapiserver/mux:go_default_library",
        "//pkg/storage/etcd/metrics:go_default_library",
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/prometheus/client_golang/prometheus",
    ],
)
