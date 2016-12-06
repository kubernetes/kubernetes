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
    srcs = ["start.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/kubernetes-discovery/pkg/apis/apiregistration/v1alpha1:go_default_library",
        "//cmd/kubernetes-discovery/pkg/apiserver:go_default_library",
        "//cmd/kubernetes-discovery/pkg/legacy:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/genericapiserver:go_default_library",
        "//pkg/genericapiserver/options:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/registry/generic:go_default_library",
        "//pkg/registry/generic/registry:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/storage/storagebackend:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//vendor:github.com/pborman/uuid",
        "//vendor:github.com/spf13/cobra",
    ],
)
