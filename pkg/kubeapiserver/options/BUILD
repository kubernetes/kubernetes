package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "authentication.go",
        "authorization.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/controller/informers:go_default_library",
        "//pkg/genericapiserver:go_default_library",
        "//pkg/genericapiserver/options:go_default_library",
        "//pkg/kubeapiserver/authenticator:go_default_library",
        "//pkg/kubeapiserver/authorizer:go_default_library",
        "//vendor:github.com/spf13/pflag",
    ],
)
