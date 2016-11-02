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
        "clientset_generated.go",
        "doc.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//cmd/libs/go2idl/client-gen/testoutput/clientset_generated/test_internalclientset:go_default_library",
        "//cmd/libs/go2idl/client-gen/testoutput/clientset_generated/test_internalclientset/typed/testgroup/internalversion:go_default_library",
        "//cmd/libs/go2idl/client-gen/testoutput/clientset_generated/test_internalclientset/typed/testgroup/internalversion/fake:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/client/testing/core:go_default_library",
        "//pkg/client/typed/discovery:go_default_library",
        "//pkg/client/typed/discovery/fake:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/watch:go_default_library",
    ],
)
