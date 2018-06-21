load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "auth.go",
        "basic.go",
        "digest.go",
        "md5crypt.go",
        "misc.go",
        "users.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/abbot/go-http-auth",
    importpath = "github.com/abbot/go-http-auth",
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
