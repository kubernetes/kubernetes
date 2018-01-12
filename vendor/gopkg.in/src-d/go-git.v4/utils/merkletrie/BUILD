load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "change.go",
        "difftree.go",
        "doc.go",
        "doubleiter.go",
        "iter.go",
    ],
    importpath = "gopkg.in/src-d/go-git.v4/utils/merkletrie",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/gopkg.in/src-d/go-git.v4/utils/merkletrie/internal/frame:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/utils/merkletrie/noder:go_default_library",
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
        "//vendor/gopkg.in/src-d/go-git.v4/utils/merkletrie/filesystem:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/utils/merkletrie/index:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/utils/merkletrie/internal/frame:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/utils/merkletrie/noder:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
