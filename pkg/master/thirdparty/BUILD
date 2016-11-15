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
    srcs = ["thirdparty.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/meta:go_default_library",
        "//pkg/api/rest:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apiserver:go_default_library",
        "//pkg/genericapiserver:go_default_library",
        "//pkg/registry/extensions/rest:go_default_library",
        "//pkg/registry/extensions/thirdpartyresourcedata:go_default_library",
        "//pkg/registry/extensions/thirdpartyresourcedata/etcd:go_default_library",
        "//pkg/registry/generic:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/storage/storagebackend:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)
