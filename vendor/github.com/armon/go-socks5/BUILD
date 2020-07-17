load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "auth.go",
        "credentials.go",
        "request.go",
        "resolver.go",
        "ruleset.go",
        "socks5.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/armon/go-socks5",
    importpath = "github.com/armon/go-socks5",
    visibility = ["//visibility:public"],
    deps = ["//vendor/golang.org/x/net/context:go_default_library"],
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
