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
        "generated.pb.go",
        "intstr.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/genericapiserver/openapi/common:go_default_library",
        "//vendor:github.com/go-openapi/spec",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/gofuzz",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["intstr_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = ["//vendor:github.com/ghodss/yaml"],
)
