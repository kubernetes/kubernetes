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
        "//pkg/util/cert:go_default_library",
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
