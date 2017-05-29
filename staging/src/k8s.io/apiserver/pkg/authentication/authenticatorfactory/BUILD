package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "delegating.go",
        "loopback.go",
        "requestheader.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/github.com/go-openapi/spec:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/authenticator:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/group:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/request/anonymous:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/request/bearertoken:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/request/headerrequest:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/request/union:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/request/x509:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/token/tokenfile:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//vendor/k8s.io/apiserver/plugin/pkg/authenticator/token/webhook:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/authentication/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/util/cert:go_default_library",
    ],
)
