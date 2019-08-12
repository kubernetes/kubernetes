package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "etcd.go",
        "strategy.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiextensions-apiserver/pkg/registry/customresourcedefinition",
    importpath = "k8s.io/apiextensions-apiserver/pkg/registry/customresourcedefinition",
    deps = [
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/validation:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/features:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/endpoints/request:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/registry/generic:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/registry/generic/registry:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/registry/rest:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/errors:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/names:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/dryrun:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
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

go_test(
    name = "go_default_test",
    srcs = ["strategy_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/validation:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/features:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/component-base/featuregate/testing:go_default_library",
        "//vendor/k8s.io/utils/pointer:go_default_library",
    ],
)
