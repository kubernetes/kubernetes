load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "serialization.go",
        "signatures.go",
        "types.go",
    ],
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/google/certificate-transparency/go/x509:go_default_library"],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//vendor/github.com/google/certificate-transparency/go/asn1:all-srcs",
        "//vendor/github.com/google/certificate-transparency/go/client:all-srcs",
        "//vendor/github.com/google/certificate-transparency/go/x509:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
