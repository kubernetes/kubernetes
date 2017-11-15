load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "conversion.go",
        "doc.go",
        "register.go",
        "types.go",
        "zz_generated.conversion.go",
        "zz_generated.deepcopy.go",
    ],
    importpath = "k8s.io/apimachinery/pkg/apis/meta/internalversion",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/apis/meta/v1alpha1:go_default_library",
        "//pkg/conversion:go_default_library",
        "//pkg/fields:go_default_library",
        "//pkg/labels:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/runtime/serializer:go_default_library",
        "//pkg/util/validation/field:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "register_test.go",
        "roundtrip_test.go",
    ],
    importpath = "k8s.io/apimachinery/pkg/apis/meta/internalversion",
    library = ":go_default_library",
    deps = [
        "//pkg/api/testing/roundtrip:go_default_library",
        "//pkg/apis/meta/fuzzer:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/util/diff:go_default_library",
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
