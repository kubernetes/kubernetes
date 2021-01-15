load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "atmostevery.go",
        "conflict.go",
        "fields.go",
        "gvkparser.go",
        "managedfields.go",
        "pathelement.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager/internal",
    importpath = "k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager/internal",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/schemaconv:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/util/proto:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/v4/fieldpath:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/v4/merge:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/v4/typed:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/v4/value:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "atmostevery_test.go",
        "conflict_test.go",
        "fields_test.go",
        "managedfields_test.go",
        "pathelement_test.go",
    ],
    data = glob(["testdata/**"]),
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/v4/fieldpath:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/v4/merge:go_default_library",
        "//vendor/sigs.k8s.io/yaml:go_default_library",
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
