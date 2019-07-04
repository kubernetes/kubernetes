load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "dispenser.go",
        "json.go",
        "lexer.go",
        "parse.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/mholt/caddy/caddyfile",
    importpath = "github.com/mholt/caddy/caddyfile",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/mholt/caddy/telemetry:go_default_library"],
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
