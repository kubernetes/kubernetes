package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_go//go:def.bzl", "go_test")

go_test(
    name = "go_default_test",
    srcs = ["cacher_test.go"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/api/apitesting:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/apis/example:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/apis/example/v1:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/features:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/cacher:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/etcd3:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/etcd3/testing:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/testing:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/value:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/component-base/featuregate/testing:go_default_library",
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
