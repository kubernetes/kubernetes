load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "ed25519.go",
        "ed25519_go113.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/crypto/ed25519",
    importpath = "golang.org/x/crypto/ed25519",
    visibility = ["//visibility:public"],
    deps = ["//vendor/golang.org/x/crypto/ed25519/internal/edwards25519:go_default_library"],
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
        "//vendor/golang.org/x/crypto/ed25519/internal/edwards25519:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
