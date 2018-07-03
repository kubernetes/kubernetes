load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "asn1.go",
        "builder.go",
        "string.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/crypto/cryptobyte",
    importpath = "golang.org/x/crypto/cryptobyte",
    visibility = ["//visibility:public"],
    deps = ["//vendor/golang.org/x/crypto/cryptobyte/asn1:go_default_library"],
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
        "//vendor/golang.org/x/crypto/cryptobyte/asn1:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
