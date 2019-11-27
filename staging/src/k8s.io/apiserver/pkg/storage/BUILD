package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "selection_predicate_test.go",
        "util_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "errors.go",
        "interfaces.go",
        "selection_predicate.go",
        "util.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/storage",
    importpath = "k8s.io/apiserver/pkg/storage",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/validation/path:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/watch:go_default_library",
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
        "//staging/src/k8s.io/apiserver/pkg/storage/cacher:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/storage/errors:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/storage/etcd3:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/storage/names:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/storage/storagebackend:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/storage/testing:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/storage/tests:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/storage/value:all-srcs",
    ],
    tags = ["automanaged"],
)
