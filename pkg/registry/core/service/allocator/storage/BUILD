package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["storage_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/registry/core/service/allocator:go_default_library",
        "//pkg/registry/registrytest:go_default_library",
        "//pkg/storage/etcd/testing:go_default_library",
        "//pkg/storage/storagebackend:go_default_library",
        "//vendor:golang.org/x/net/context",
    ],
)

go_library(
    name = "go_default_library",
    srcs = ["storage.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors/storage:go_default_library",
        "//pkg/registry/core/rangeallocation:go_default_library",
        "//pkg/registry/core/service/allocator:go_default_library",
        "//pkg/registry/generic:go_default_library",
        "//pkg/storage:go_default_library",
        "//pkg/storage/storagebackend:go_default_library",
        "//vendor:golang.org/x/net/context",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
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
)
