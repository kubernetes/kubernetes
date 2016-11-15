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
    srcs = ["oidc.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/auth/user:go_default_library",
        "//pkg/util/cert:go_default_library",
        "//pkg/util/net:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//vendor:github.com/coreos/go-oidc/jose",
        "//vendor:github.com/coreos/go-oidc/oidc",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["oidc_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/auth/user:go_default_library",
        "//plugin/pkg/auth/authenticator/token/oidc/testing:go_default_library",
        "//vendor:github.com/coreos/go-oidc/jose",
        "//vendor:github.com/coreos/go-oidc/oidc",
    ],
)
