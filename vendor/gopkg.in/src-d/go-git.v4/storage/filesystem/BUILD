load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "config.go",
        "deltaobject.go",
        "index.go",
        "module.go",
        "object.go",
        "reference.go",
        "shallow.go",
        "storage.go",
    ],
    importpath = "gopkg.in/src-d/go-git.v4/storage/filesystem",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/gopkg.in/src-d/go-billy.v3:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/config:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/cache:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/format/idxfile:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/format/index:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/format/objfile:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/format/packfile:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/storer:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/storage:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/storage/filesystem/internal/dotgit:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/storage/memory:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/storage/filesystem/internal/dotgit:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
