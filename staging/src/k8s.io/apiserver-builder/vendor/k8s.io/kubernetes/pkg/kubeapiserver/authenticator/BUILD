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
        "//pkg/serviceaccount:go_default_library",
        "//vendor:github.com/go-openapi/spec",
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticatorfactory",
        "//vendor:k8s.io/apiserver/pkg/authentication/group",
        "//vendor:k8s.io/apiserver/pkg/authentication/request/anonymous",
        "//vendor:k8s.io/apiserver/pkg/authentication/request/bearertoken",
        "//vendor:k8s.io/apiserver/pkg/authentication/request/headerrequest",
        "//vendor:k8s.io/apiserver/pkg/authentication/request/union",
        "//vendor:k8s.io/apiserver/pkg/authentication/request/x509",
        "//vendor:k8s.io/apiserver/pkg/authentication/token/tokenfile",
        "//vendor:k8s.io/apiserver/plugin/pkg/authenticator/password/keystone",
        "//vendor:k8s.io/apiserver/plugin/pkg/authenticator/password/passwordfile",
        "//vendor:k8s.io/apiserver/plugin/pkg/authenticator/request/basicauth",
        "//vendor:k8s.io/apiserver/plugin/pkg/authenticator/token/anytoken",
        "//vendor:k8s.io/apiserver/plugin/pkg/authenticator/token/oidc",
        "//vendor:k8s.io/apiserver/plugin/pkg/authenticator/token/webhook",
        "//vendor:k8s.io/client-go/plugin/pkg/client/auth/gcp",
        "//vendor:k8s.io/client-go/plugin/pkg/client/auth/oidc",
        "//vendor:k8s.io/client-go/util/cert",
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
