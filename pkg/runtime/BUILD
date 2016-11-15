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
        "codec.go",
        "codec_check.go",
        "conversion.go",
        "doc.go",
        "embedded.go",
        "error.go",
        "extension.go",
        "generated.pb.go",
        "helper.go",
        "interfaces.go",
        "register.go",
        "scheme.go",
        "scheme_builder.go",
        "swagger_doc_generator.go",
        "types.go",
        "types_proto.go",
        "unstructured.go",
        "zz_generated.deepcopy.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/meta/metatypes:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/conversion:go_default_library",
        "//pkg/conversion/queryparams:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/errors:go_default_library",
        "//pkg/util/json:go_default_library",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["swagger_doc_generator_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)

go_test(
    name = "go_default_xtest",
    srcs = [
        "conversion_test.go",
        "embedded_test.go",
        "extension_test.go",
        "helper_test.go",
        "scheme_test.go",
        "unstructured_test.go",
        "unversioned_test.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/meta:go_default_library",
        "//pkg/api/meta/metatypes:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/api/validation:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/conversion:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/serializer:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/diff:go_default_library",
        "//vendor:github.com/google/gofuzz",
        "//vendor:github.com/spf13/pflag",
    ],
)
