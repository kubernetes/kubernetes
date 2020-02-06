load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "oauth2.go",
        "token.go",
        "transport.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/oauth2/internal",
    importpath = "golang.org/x/oauth2/internal",
    visibility = ["//vendor/golang.org/x/oauth2:__subpackages__"],
    deps = ["//vendor/golang.org/x/net/context/ctxhttp:go_default_library"],
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
