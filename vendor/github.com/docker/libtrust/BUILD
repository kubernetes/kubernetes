load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "certificates.go",
        "doc.go",
        "ec_key.go",
        "filter.go",
        "hash.go",
        "jsonsign.go",
        "key.go",
        "key_files.go",
        "key_manager.go",
        "rsa_key.go",
        "util.go",
    ],
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
