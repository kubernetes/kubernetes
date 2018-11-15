load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "dirent.go",
        "doc.go",
        "readdir.go",
        "readdir_unix.go",
        "readdir_windows.go",
        "walk.go",
        "withFileno.go",
        "withIno.go",
        "withNamlen.go",
        "withoutNamlen.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/karrick/godirwalk",
    importpath = "github.com/karrick/godirwalk",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/pkg/errors:go_default_library"],
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
