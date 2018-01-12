load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["client.go"],
    importpath = "gopkg.in/src-d/go-git.v4/plumbing/transport/client",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/transport:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/transport/file:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/transport/git:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/transport/http:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/transport/ssh:go_default_library",
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
