load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "backoff.go",
        "client.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/google/certificate-transparency-go/jsonclient",
    importpath = "github.com/google/certificate-transparency-go/jsonclient",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/google/certificate-transparency-go:go_default_library",
        "//vendor/github.com/google/certificate-transparency-go/x509:go_default_library",
        "//vendor/golang.org/x/net/context/ctxhttp:go_default_library",
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
