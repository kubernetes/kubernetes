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
        "//pkg/auth/authenticator/bearertoken:go_default_library",
        "//pkg/auth/group:go_default_library",
        "//pkg/util/cert:go_default_library",
        "//plugin/pkg/auth/authenticator/request/anonymous:go_default_library",
        "//plugin/pkg/auth/authenticator/request/headerrequest:go_default_library",
        "//plugin/pkg/auth/authenticator/request/union:go_default_library",
        "//plugin/pkg/auth/authenticator/request/x509:go_default_library",
        "//plugin/pkg/auth/authenticator/token/tokenfile:go_default_library",
        "//plugin/pkg/auth/authenticator/token/webhook:go_default_library",
        "//vendor:github.com/go-openapi/spec",
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/client-go/kubernetes/typed/authentication/v1beta1",
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
