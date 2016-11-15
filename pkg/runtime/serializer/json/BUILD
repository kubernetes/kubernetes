package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "json.go",
        "meta.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/unversioned:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/serializer/recognizer:go_default_library",
        "//pkg/util/framer:go_default_library",
        "//pkg/util/yaml:go_default_library",
        "//vendor:github.com/ghodss/yaml",
        "//vendor:github.com/ugorji/go/codec",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["meta_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)

go_test(
    name = "go_default_xtest",
    srcs = ["json_test.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/unversioned:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/serializer/json:go_default_library",
        "//pkg/util/diff:go_default_library",
    ],
)
