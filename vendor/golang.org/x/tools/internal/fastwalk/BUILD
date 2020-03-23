load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "fastwalk.go",
        "fastwalk_dirent_fileno.go",
        "fastwalk_dirent_ino.go",
        "fastwalk_dirent_namlen_bsd.go",
        "fastwalk_dirent_namlen_linux.go",
        "fastwalk_portable.go",
        "fastwalk_unix.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/tools/internal/fastwalk",
    importpath = "golang.org/x/tools/internal/fastwalk",
    visibility = ["//vendor/golang.org/x/tools:__subpackages__"],
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
