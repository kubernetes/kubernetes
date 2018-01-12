load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "common.go",
        "receive_pack.go",
        "upload_pack.go",
    ],
    importpath = "gopkg.in/src-d/go-git.v4/plumbing/transport/http",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/format/pktline:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/protocol/packp:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/protocol/packp/capability:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/protocol/packp/sideband:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/transport:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/transport/internal/common:go_default_library",
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
