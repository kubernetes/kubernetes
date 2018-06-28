load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bzr.go",
        "errors.go",
        "git.go",
        "hg.go",
        "repo.go",
        "svn.go",
        "vcs_local_lookup.go",
        "vcs_remote_lookup.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/Masterminds/vcs",
    importpath = "github.com/Masterminds/vcs",
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
