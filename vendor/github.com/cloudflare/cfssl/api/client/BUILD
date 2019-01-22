load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "api.go",
        "client.go",
        "group.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/cloudflare/cfssl/api/client",
    importpath = "github.com/cloudflare/cfssl/api/client",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/cloudflare/cfssl/api:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/auth:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/errors:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/info:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/log:go_default_library",
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
