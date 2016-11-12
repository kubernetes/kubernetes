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
        "context.go",
        "conversion.go",
        "defaults.go",
        "doc.go",
        "field_constants.go",
        "generate.go",
        "helpers.go",
        "mapper.go",
        "meta.go",
        "ref.go",
        "register.go",
        "requestcontext.go",
        "resource_helpers.go",
        "types.generated.go",
        "types.go",
        "zz_generated.deepcopy.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "generate_test.go",
        "helpers_test.go",
        "ref_test.go",
        "resource_helpers_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)

go_test(
    name = "go_default_xtest",
    srcs = [
        "context_test.go",
        "conversion_test.go",
        "copy_test.go",
        "deep_copy_test.go",
        "defaulting_test.go",
        "meta_test.go",
        "serialization_proto_test.go",
        "serialization_test.go",
    ],
    tags = ["automanaged"],
    deps = [],
)
