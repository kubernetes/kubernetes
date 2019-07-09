load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "dispenser.go",
        "json.go",
        "lexer.go",
        "parse.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/caddyserver/caddy/caddyfile",
    importpath = "github.com/caddyserver/caddy/caddyfile",
    visibility = ["//visibility:public"],
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
