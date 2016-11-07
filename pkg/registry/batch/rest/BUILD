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
    srcs = ["storage_batch.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/rest:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/apis/batch:go_default_library",
        "//pkg/apis/batch/v1:go_default_library",
        "//pkg/apis/batch/v2alpha1:go_default_library",
        "//pkg/genericapiserver:go_default_library",
        "//pkg/registry/batch/cronjob/etcd:go_default_library",
        "//pkg/registry/batch/job/etcd:go_default_library",
    ],
)
