load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "algorithm.go",
        "prune.go",
        "surroundingobject.go",
        "validation.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiextensions-apiserver/pkg/apiserver/schema/defaulting",
    importpath = "k8s.io/apiextensions-apiserver/pkg/apiserver/schema/defaulting",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apiserver/schema:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apiserver/schema/objectmeta:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apiserver/schema/pruning:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apiserver/validation:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//vendor/github.com/go-openapi/strfmt:go_default_library",
        "//vendor/github.com/go-openapi/validate:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["algorithm_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apiserver/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/json:go_default_library",
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
    visibility = ["//visibility:public"],
)
