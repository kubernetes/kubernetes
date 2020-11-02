load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "base64.go",
        "bcrypt.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/crypto/bcrypt",
    importpath = "golang.org/x/crypto/bcrypt",
    visibility = ["//visibility:public"],
    deps = ["//vendor/golang.org/x/crypto/blowfish:go_default_library"],
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
