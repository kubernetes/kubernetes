package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["kubelet_client.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
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
    data = ["//pkg/client/testdata"],
    library = ":go_default_library",
    tags = [
        "automanaged",
    ],
    deps = [
        "//pkg/client/clientset_generated/clientset/typed/core/v1:go_default_library",
        "//pkg/client/restclient:go_default_library",
    ],
)
