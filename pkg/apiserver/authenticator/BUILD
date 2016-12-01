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
        "builtin.go",
        "delegating.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/auth/authenticator:go_default_library",
        "//pkg/auth/authenticator/bearertoken:go_default_library",
        "//pkg/auth/group:go_default_library",
        "//pkg/auth/user:go_default_library",
        "//pkg/client/clientset_generated/release_1_5/typed/authentication/v1beta1:go_default_library",
        "//pkg/serviceaccount:go_default_library",
        "//pkg/util/cert:go_default_library",
        "//plugin/pkg/auth/authenticator/password/keystone:go_default_library",
        "//plugin/pkg/auth/authenticator/password/passwordfile:go_default_library",
        "//plugin/pkg/auth/authenticator/request/anonymous:go_default_library",
        "//plugin/pkg/auth/authenticator/request/basicauth:go_default_library",
        "//plugin/pkg/auth/authenticator/request/headerrequest:go_default_library",
        "//plugin/pkg/auth/authenticator/request/union:go_default_library",
        "//plugin/pkg/auth/authenticator/request/x509:go_default_library",
        "//plugin/pkg/auth/authenticator/token/anytoken:go_default_library",
        "//plugin/pkg/auth/authenticator/token/oidc:go_default_library",
        "//plugin/pkg/auth/authenticator/token/tokenfile:go_default_library",
        "//plugin/pkg/auth/authenticator/token/webhook:go_default_library",
        "//vendor:github.com/go-openapi/spec",
    ],
)
