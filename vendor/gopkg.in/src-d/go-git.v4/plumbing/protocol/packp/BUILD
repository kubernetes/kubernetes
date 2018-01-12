load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "advrefs.go",
        "advrefs_decode.go",
        "advrefs_encode.go",
        "common.go",
        "doc.go",
        "report_status.go",
        "shallowupd.go",
        "srvresp.go",
        "ulreq.go",
        "ulreq_decode.go",
        "ulreq_encode.go",
        "updreq.go",
        "updreq_decode.go",
        "updreq_encode.go",
        "uppackreq.go",
        "uppackresp.go",
    ],
    importpath = "gopkg.in/src-d/go-git.v4/plumbing/protocol/packp",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/format/pktline:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/protocol/packp/capability:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/protocol/packp/sideband:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/storer:go_default_library",
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
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/protocol/packp/capability:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/protocol/packp/sideband:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
