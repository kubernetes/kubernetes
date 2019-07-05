load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "jose.go",
        "jwks.go",
        "oidc.go",
        "verify.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/coreos/go-oidc",
    importpath = "github.com/coreos/go-oidc",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/pquerna/cachecontrol:go_default_library",
        "//vendor/golang.org/x/oauth2:go_default_library",
        "//vendor/gopkg.in/square/go-jose.v2:go_default_library",
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
