load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "direct.go",
        "per_host.go",
        "proxy.go",
        "socks5.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/net/proxy",
    importpath = "golang.org/x/net/proxy",
    visibility = ["//visibility:public"],
    deps = ["//vendor/golang.org/x/net/internal/socks:go_default_library"],
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
