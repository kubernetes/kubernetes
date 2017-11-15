load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["validation.go"],
    importpath = "k8s.io/apimachinery/pkg/apis/meta/v1/validation",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/util/validation:go_default_library",
        "//pkg/util/validation/field:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["validation_test.go"],
    importpath = "k8s.io/apimachinery/pkg/apis/meta/v1/validation",
    library = ":go_default_library",
    deps = ["//pkg/util/validation/field:go_default_library"],
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
