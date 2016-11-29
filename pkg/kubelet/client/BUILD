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
    srcs = ["kubelet_client.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/transport:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/net:go_default_library",
        "//pkg/util/node:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["kubelet_client_test.go"],
    library = "go_default_library",
    tags = [
        "automanaged",
        "skip",
    ],
    deps = [
        "//pkg/client/clientset_generated/release_1_5/typed/core/v1:go_default_library",
        "//pkg/client/restclient:go_default_library",
    ],
)
