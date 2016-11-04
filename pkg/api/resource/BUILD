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
        "amount.go",
        "generated.pb.go",
        "math.go",
        "quantity.go",
        "quantity_proto.go",
        "scale_int.go",
        "suffix.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/genericapiserver/openapi/common:go_default_library",
        "//vendor:github.com/go-openapi/spec",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/spf13/pflag",
        "//vendor:gopkg.in/inf.v0",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "amount_test.go",
        "math_test.go",
        "quantity_test.go",
        "scale_int_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/google/gofuzz",
        "//vendor:github.com/spf13/pflag",
        "//vendor:gopkg.in/inf.v0",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = ["quantity_example_test.go"],
    tags = ["automanaged"],
    deps = ["//pkg/api/resource:go_default_library"],
)
