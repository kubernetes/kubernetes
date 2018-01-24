load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["subdir.go"],
    importpath = "k8s.io/kubernetes/cmd/staginghub/override",
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/go-git-http:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/object:go_default_library",
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
