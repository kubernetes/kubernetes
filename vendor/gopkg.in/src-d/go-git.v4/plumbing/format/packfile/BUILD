load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "common.go",
        "decoder.go",
        "delta_selector.go",
        "diff_delta.go",
        "doc.go",
        "encoder.go",
        "error.go",
        "index.go",
        "object_pack.go",
        "patch_delta.go",
        "scanner.go",
    ],
    importpath = "gopkg.in/src-d/go-git.v4/plumbing/format/packfile",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/cache:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/format/idxfile:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/storer:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/utils/binary:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/utils/ioutil:go_default_library",
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
