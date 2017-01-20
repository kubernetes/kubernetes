package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["config.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/genericapiserver/authenticator:go_default_library",
        "//pkg/serviceaccount:go_default_library",
        "//pkg/util/cert:go_default_library",
        "//plugin/pkg/auth/authenticator/password/keystone:go_default_library",
        "//plugin/pkg/auth/authenticator/password/passwordfile:go_default_library",
        "//plugin/pkg/auth/authenticator/request/basicauth:go_default_library",
        "//plugin/pkg/auth/authenticator/token/anytoken:go_default_library",
        "//plugin/pkg/auth/authenticator/token/oidc:go_default_library",
        "//plugin/pkg/auth/authenticator/token/webhook:go_default_library",
        "//vendor:github.com/go-openapi/spec",
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/group",
        "//vendor:k8s.io/apiserver/pkg/authentication/request/anonymous",
        "//vendor:k8s.io/apiserver/pkg/authentication/request/bearertoken",
        "//vendor:k8s.io/apiserver/pkg/authentication/request/headerrequest",
        "//vendor:k8s.io/apiserver/pkg/authentication/request/union",
        "//vendor:k8s.io/apiserver/pkg/authentication/request/x509",
        "//vendor:k8s.io/apiserver/pkg/authentication/token/tokenfile",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/client-go/plugin/pkg/client/auth/gcp",
        "//vendor:k8s.io/client-go/plugin/pkg/client/auth/oidc",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
