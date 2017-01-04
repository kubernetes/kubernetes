package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["oidc.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/util/cert:go_default_library",
        "//pkg/util/net:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//vendor:github.com/coreos/go-oidc/jose",
        "//vendor:github.com/coreos/go-oidc/oidc",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["oidc_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//plugin/pkg/auth/authenticator/token/oidc/testing:go_default_library",
        "//vendor:github.com/coreos/go-oidc/jose",
        "//vendor:github.com/coreos/go-oidc/oidc",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
    ],
)
