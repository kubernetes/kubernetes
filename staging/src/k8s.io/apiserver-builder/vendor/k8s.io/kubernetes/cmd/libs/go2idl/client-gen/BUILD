package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "client-gen",
    library = ":go_default_library",
    tags = ["automanaged"],
)

go_library(
    name = "go_default_library",
    srcs = ["main.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/libs/go2idl/client-gen/args:go_default_library",
        "//cmd/libs/go2idl/client-gen/generators:go_default_library",
        "//cmd/libs/go2idl/client-gen/types:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/spf13/pflag",
        "//vendor:k8s.io/gengo/args",
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
        "//cmd/libs/go2idl/client-gen/args:all-srcs",
        "//cmd/libs/go2idl/client-gen/generators:all-srcs",
        "//cmd/libs/go2idl/client-gen/test_apis/testgroup:all-srcs",
        "//cmd/libs/go2idl/client-gen/testoutput/clientset_generated/test_internalclientset:all-srcs",
        "//cmd/libs/go2idl/client-gen/types:all-srcs",
    ],
    tags = ["automanaged"],
)
