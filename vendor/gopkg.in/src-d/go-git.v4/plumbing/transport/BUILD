load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["common.go"],
    importpath = "gopkg.in/src-d/go-git.v4/plumbing/transport",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/protocol/packp:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/protocol/packp/capability:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/transport/client:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/transport/file:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/transport/git:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/transport/http:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/transport/internal/common:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/transport/server:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/transport/ssh:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
