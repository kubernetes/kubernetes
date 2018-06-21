load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "fastwalk.go",
        "fastwalk_dirent_fileno.go",
        "fastwalk_dirent_ino.go",
        "fastwalk_portable.go",
        "fastwalk_unix.go",
        "fix.go",
        "imports.go",
        "sortimports.go",
        "zstdlib.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/tools/imports",
    importpath = "golang.org/x/tools/imports",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/tools/go/ast/astutil:go_default_library",
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
