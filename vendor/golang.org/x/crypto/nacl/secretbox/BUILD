load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["secretbox.go"],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/crypto/nacl/secretbox",
    importpath = "golang.org/x/crypto/nacl/secretbox",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/crypto/internal/subtle:go_default_library",
        "//vendor/golang.org/x/crypto/poly1305:go_default_library",
        "//vendor/golang.org/x/crypto/salsa20/salsa:go_default_library",
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
