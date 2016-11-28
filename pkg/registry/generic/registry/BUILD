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
        "decorated_watcher.go",
        "doc.go",
        "storage_factory.go",
        "store.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/errors/storage:go_default_library",
        "//pkg/api/meta:go_default_library",
        "//pkg/api/rest:go_default_library",
        "//pkg/api/validation/path:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/fields:go_default_library",
        "//pkg/labels:go_default_library",
        "//pkg/registry/generic:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/storage:go_default_library",
        "//pkg/storage/etcd:go_default_library",
        "//pkg/storage/storagebackend:go_default_library",
        "//pkg/storage/storagebackend/factory:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/util/validation/field:go_default_library",
        "//pkg/version:go_default_library",
        "//pkg/watch:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/context",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "decorated_watcher_test.go",
        "store_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/meta:go_default_library",
        "//pkg/api/rest:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/fields:go_default_library",
        "//pkg/labels:go_default_library",
        "//pkg/registry/core/pod:go_default_library",
        "//pkg/registry/generic:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/selection:go_default_library",
        "//pkg/storage:go_default_library",
        "//pkg/storage/etcd:go_default_library",
        "//pkg/storage/etcd/testing:go_default_library",
        "//pkg/storage/storagebackend/factory:go_default_library",
        "//pkg/storage/testing:go_default_library",
        "//pkg/util/validation/field:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//pkg/watch:go_default_library",
    ],
)
