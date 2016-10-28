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
    srcs = ["install.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/libs/go2idl/client-gen/test_apis/testgroup:go_default_library",
        "//cmd/libs/go2idl/client-gen/test_apis/testgroup/v1:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/meta:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/apimachinery:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)
