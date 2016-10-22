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
    srcs = ["etcd.go"],
    tags = ["automanaged"],
    deps = [
        "//federation/apis/federation:go_default_library",
        "//federation/registry/cluster:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/rest:go_default_library",
        "//pkg/registry/generic:go_default_library",
        "//pkg/registry/generic/registry:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/storage:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["etcd_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//federation/apis/federation:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/fields:go_default_library",
        "//pkg/labels:go_default_library",
        "//pkg/registry/generic:go_default_library",
        "//pkg/registry/registrytest:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/storage/etcd/testing:go_default_library",
    ],
)
