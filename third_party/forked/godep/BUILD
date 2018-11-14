load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

licenses(["notice"])

go_library(
    name = "go_default_library",
    srcs = [
        "dep.go",
        "diff.go",
        "doc.go",
        "errors.go",
        "get.go",
        "go.go",
        "godepfile.go",
        "license.go",
        "list.go",
        "main.go",
        "msg.go",
        "path.go",
        "pkg.go",
        "restore.go",
        "rewrite.go",
        "save.go",
        "update.go",
        "util.go",
        "vcs.go",
        "version.go",
    ],
    importpath = "k8s.io/kubernetes/third_party/forked/godep",
    visibility = ["//visibility:private"],
    deps = [
        "//vendor/github.com/kr/fs:go_default_library",
        "//vendor/github.com/kr/pretty:go_default_library",
        "//vendor/github.com/pmezard/go-difflib/difflib:go_default_library",
        "//vendor/golang.org/x/tools/go/vcs:go_default_library",
    ],
)

go_binary(
    name = "godep",
    embed = [":go_default_library"],
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
