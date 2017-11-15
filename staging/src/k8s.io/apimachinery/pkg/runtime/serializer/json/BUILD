load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "json.go",
        "meta.go",
    ],
    importpath = "k8s.io/apimachinery/pkg/runtime/serializer/json",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/runtime/serializer/recognizer:go_default_library",
        "//pkg/util/framer:go_default_library",
        "//pkg/util/yaml:go_default_library",
        "//vendor/github.com/ghodss/yaml:go_default_library",
        "//vendor/github.com/json-iterator/go:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["meta_test.go"],
    importpath = "k8s.io/apimachinery/pkg/runtime/serializer/json",
    library = ":go_default_library",
)

go_test(
    name = "go_default_xtest",
    srcs = ["json_test.go"],
    importpath = "k8s.io/apimachinery/pkg/runtime/serializer/json_test",
    deps = [
        ":go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
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
