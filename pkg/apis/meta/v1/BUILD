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
        "doc.go",
        "duration.go",
        "generated.pb.go",
        "group_version.go",
        "helpers.go",
        "meta.go",
        "register.go",
        "time.go",
        "time_proto.go",
        "types.go",
        "types_swagger_doc_generated.go",
        "well_known_labels.go",
        "zz_generated.deepcopy.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/conversion:go_default_library",
        "//pkg/genericapiserver/openapi/common:go_default_library",
        "//pkg/labels:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/selection:go_default_library",
        "//vendor:github.com/go-openapi/spec",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/gogo/protobuf/sortkeys",
        "//vendor:github.com/google/gofuzz",
        "//vendor:github.com/ugorji/go/codec",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "duration_test.go",
        "group_version_test.go",
        "helpers_test.go",
        "time_test.go",
        "types_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/labels:go_default_library",
        "//vendor:github.com/ghodss/yaml",
        "//vendor:github.com/ugorji/go/codec",
    ],
)
