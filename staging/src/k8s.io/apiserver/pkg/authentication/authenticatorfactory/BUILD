load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "delegating.go",
        "loopback.go",
        "requestheader.go",
    ],
    importpath = "k8s.io/apiserver/pkg/authentication/authenticatorfactory",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/authentication/authenticator:go_default_library",
        "//pkg/authentication/group:go_default_library",
        "//pkg/authentication/request/anonymous:go_default_library",
        "//pkg/authentication/request/bearertoken:go_default_library",
        "//pkg/authentication/request/headerrequest:go_default_library",
        "//pkg/authentication/request/union:go_default_library",
        "//pkg/authentication/request/websocket:go_default_library",
        "//pkg/authentication/request/x509:go_default_library",
        "//pkg/authentication/token/tokenfile:go_default_library",
        "//pkg/authentication/user:go_default_library",
        "//plugin/pkg/authenticator/token/webhook:go_default_library",
        "//vendor/github.com/go-openapi/spec:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/authentication/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/util/cert:go_default_library",
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
    visibility = ["//visibility:public"],
)
