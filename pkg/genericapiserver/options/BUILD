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
        "authenticator.go",
        "doc.go",
        "etcd.go",
        "server_run_options.go",
        "serving.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/admission:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/apiserver/authenticator:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/storage/storagebackend:go_default_library",
        "//pkg/util/config:go_default_library",
        "//pkg/util/net:go_default_library",
        "//vendor:github.com/spf13/pflag",
    ],
)
