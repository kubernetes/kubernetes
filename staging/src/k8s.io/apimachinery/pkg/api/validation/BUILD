load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "generic.go",
        "objectmeta.go",
    ],
    importpath = "k8s.io/apimachinery/pkg/api/validation",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/api/equality:go_default_library",
        "//pkg/api/meta:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/apis/meta/v1/validation:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/validation:go_default_library",
        "//pkg/util/validation/field:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["objectmeta_test.go"],
    importpath = "k8s.io/apimachinery/pkg/api/validation",
    library = ":go_default_library",
    deps = [
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/util/validation/field:go_default_library",
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
        "//staging/src/k8s.io/apimachinery/pkg/api/validation/path:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
