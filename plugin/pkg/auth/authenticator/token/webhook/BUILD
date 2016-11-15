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
    srcs = ["webhook.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/unversioned:go_default_library",
        "//pkg/apis/authentication:go_default_library",
        "//pkg/apis/authentication/install:go_default_library",
        "//pkg/apis/authentication/v1beta1:go_default_library",
        "//pkg/auth/authenticator:go_default_library",
        "//pkg/auth/user:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/authentication/internalversion:go_default_library",
        "//pkg/util/cache:go_default_library",
        "//plugin/pkg/webhook:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "certs_test.go",
        "webhook_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/unversioned:go_default_library",
        "//pkg/apis/authentication/v1beta1:go_default_library",
        "//pkg/auth/user:go_default_library",
        "//pkg/client/unversioned/clientcmd/api/v1:go_default_library",
    ],
)
