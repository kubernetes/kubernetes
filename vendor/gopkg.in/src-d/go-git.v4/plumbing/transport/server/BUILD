load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "loader.go",
        "server.go",
    ],
    importpath = "gopkg.in/src-d/go-git.v4/plumbing/transport/server",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/gopkg.in/src-d/go-billy.v3:go_default_library",
        "//vendor/gopkg.in/src-d/go-billy.v3/osfs:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/format/packfile:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/protocol/packp:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/protocol/packp/capability:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/revlist:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/storer:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/transport:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/storage/filesystem:go_default_library",
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
