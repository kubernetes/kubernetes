load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "debug_release.go",
        "dirent.go",
        "doc.go",
        "inoWithFileno.go",
        "inoWithIno.go",
        "modeType.go",
        "modeTypeWithType.go",
        "modeTypeWithoutType.go",
        "nameWithNamlen.go",
        "nameWithoutNamlen.go",
        "readdir.go",
        "readdir_unix.go",
        "readdir_windows.go",
        "reclenFromNamlen.go",
        "reclenFromReclen.go",
        "scandir_unix.go",
        "scandir_windows.go",
        "scanner.go",
        "walk.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/karrick/godirwalk",
    importpath = "github.com/karrick/godirwalk",
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
