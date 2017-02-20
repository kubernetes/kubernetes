package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["apiserver.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/libs/go2idl/client-gen/test_apis/testgroup/install:go_default_library",
        "//cmd/libs/go2idl/client-gen/test_apis/testgroup/v1:go_default_library",
        "//examples/apiserver/rest:go_default_library",
        "//pkg/kubeapiserver/options:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//examples/apiserver/rest:all-srcs",
        "//examples/apiserver/server:all-srcs",
    ],
    tags = ["automanaged"],
)
