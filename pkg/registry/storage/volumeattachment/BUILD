load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "strategy.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/registry/storage/volumeattachment",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/apis/storage:go_default_library",
        "//pkg/apis/storage/validation:go_default_library",
        "//staging/src/k8s.io/api/storage/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/endpoints/request:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/names:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["strategy_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/apis/storage:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/endpoints/request:go_default_library",
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
        "//pkg/registry/storage/volumeattachment/storage:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
