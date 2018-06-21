load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "signature.go",
        "tls.go",
        "types.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/google/certificate-transparency-go/tls",
    importpath = "github.com/google/certificate-transparency-go/tls",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/google/certificate-transparency-go/asn1:go_default_library"],
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
