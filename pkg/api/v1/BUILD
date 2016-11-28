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
        "conversion.go",
        "defaults.go",
        "doc.go",
        "generate.go",
        "generated.pb.go",
        "helpers.go",
        "meta.go",
        "ref.go",
        "register.go",
        "resource_helpers.go",
        "types.generated.go",
        "types.go",
        "types_swagger_doc_generated.go",
        "zz_generated.conversion.go",
        "zz_generated.deepcopy.go",
        "zz_generated.defaults.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/meta:go_default_library",
        "//pkg/api/meta/metatypes:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/conversion:go_default_library",
        "//pkg/fields:go_default_library",
        "//pkg/labels:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/selection:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util:go_default_library",
        "//pkg/util/intstr:go_default_library",
        "//pkg/util/parsers:go_default_library",
        "//pkg/util/rand:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/validation/field:go_default_library",
        "//pkg/watch/versioned:go_default_library",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/gogo/protobuf/sortkeys",
        "//vendor:github.com/ugorji/go/codec",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = [
        "backward_compatibility_test.go",
        "conversion_test.go",
        "defaults_test.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/api/testing/compat:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/api/validation:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/diff:go_default_library",
        "//pkg/util/intstr:go_default_library",
        "//pkg/util/validation/field:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "helpers_test.go",
        "resource_helpers_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/labels:go_default_library",
    ],
)
