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
        "//vendor:github.com/coreos/go-oidc/jose",
        "//vendor:github.com/coreos/go-oidc/oidc",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
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

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//plugin/pkg/auth/authenticator/token/oidc/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
