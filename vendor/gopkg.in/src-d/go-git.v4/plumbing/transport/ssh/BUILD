load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "auth_method.go",
        "common.go",
    ],
    importpath = "gopkg.in/src-d/go-git.v4/plumbing/transport/ssh",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/mitchellh/go-homedir:go_default_library",
        "//vendor/github.com/xanzy/ssh-agent:go_default_library",
        "//vendor/golang.org/x/crypto/ssh:go_default_library",
        "//vendor/golang.org/x/crypto/ssh/knownhosts:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/transport:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/transport/internal/common:go_default_library",
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
