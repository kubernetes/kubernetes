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
        "//pkg/api:go_default_library",
        "//pkg/api/rest:go_default_library",
        "//pkg/genericapiserver:go_default_library",
        "//pkg/genericapiserver/authorizer:go_default_library",
        "//pkg/genericapiserver/options:go_default_library",
        "//pkg/kubeapiserver/options:go_default_library",
        "//pkg/registry/generic:go_default_library",
        "//pkg/storage/storagebackend:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
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
