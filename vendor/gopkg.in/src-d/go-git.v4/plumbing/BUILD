load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "error.go",
        "hash.go",
        "memory.go",
        "object.go",
        "reference.go",
        "revision.go",
    ],
    importpath = "gopkg.in/src-d/go-git.v4/plumbing",
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
    srcs = [
        ":package-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/cache:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/filemode:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/format/config:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/format/diff:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/format/gitignore:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/format/idxfile:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/format/index:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/format/objfile:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/format/packfile:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/format/pktline:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/object:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/protocol/packp:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/revlist:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/storer:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/transport:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
