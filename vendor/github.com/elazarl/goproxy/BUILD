load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "actions.go",
        "certs.go",
        "chunked.go",
        "counterecryptor.go",
        "ctx.go",
        "dispatcher.go",
        "doc.go",
        "https.go",
        "proxy.go",
        "responses.go",
        "signer.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/elazarl/goproxy",
    importpath = "github.com/elazarl/goproxy",
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
