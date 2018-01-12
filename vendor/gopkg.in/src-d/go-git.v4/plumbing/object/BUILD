load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "blob.go",
        "change.go",
        "change_adaptor.go",
        "commit.go",
        "commit_walker.go",
        "difftree.go",
        "file.go",
        "object.go",
        "patch.go",
        "tag.go",
        "tree.go",
        "treenoder.go",
    ],
    importpath = "gopkg.in/src-d/go-git.v4/plumbing/object",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/sergi/go-diff/diffmatchpatch:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/filemode:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/format/diff:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/storer:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/utils/binary:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/utils/diff:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/utils/ioutil:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/utils/merkletrie:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
