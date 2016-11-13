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
)

go_test(
    name = "go_default_test",
    srcs = ["etcd_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/registry/core/service/allocator:go_default_library",
        "//pkg/registry/core/service/allocator/etcd:go_default_library",
        "//pkg/registry/core/service/ipallocator:go_default_library",
        "//pkg/registry/generic:go_default_library",
        "//pkg/registry/registrytest:go_default_library",
        "//pkg/storage:go_default_library",
        "//pkg/storage/etcd/testing:go_default_library",
        "//pkg/storage/storagebackend/factory:go_default_library",
        "//vendor:golang.org/x/net/context",
    ],
)
