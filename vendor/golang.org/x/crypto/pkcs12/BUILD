load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bmp-string.go",
        "crypto.go",
        "errors.go",
        "mac.go",
        "pbkdf.go",
        "pkcs12.go",
        "safebags.go",
    ],
    visibility = ["//visibility:public"],
    deps = ["//vendor/golang.org/x/crypto/pkcs12/internal/rc2:go_default_library"],
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
        "//vendor/golang.org/x/crypto/pkcs12/internal/rc2:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
