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
    srcs = ["options.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/unversioned:go_default_library",
        "//pkg/apis/componentconfig:go_default_library",
        "//pkg/client/leaderelection:go_default_library",
        "//pkg/master/ports:go_default_library",
        "//pkg/util/config:go_default_library",
        "//vendor:github.com/spf13/pflag",
    ],
)
