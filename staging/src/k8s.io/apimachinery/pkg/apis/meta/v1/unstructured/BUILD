load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "helpers.go",
        "unstructured.go",
        "unstructured_list.go",
        "zz_generated.deepcopy.go",
    ],
    importpath = "k8s.io/apimachinery/pkg/apis/meta/v1/unstructured",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/conversion/unstructured:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/json:go_default_library",
        "//pkg/util/runtime:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "helpers_test.go",
        "unstructured_list_test.go",
    ],
    importpath = "k8s.io/apimachinery/pkg/apis/meta/v1/unstructured",
    library = ":go_default_library",
    deps = [
        "//pkg/apis/meta/v1:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/github.com/stretchr/testify/require:go_default_library",
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
