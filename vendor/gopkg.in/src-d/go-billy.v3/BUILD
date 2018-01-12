load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["fs.go"],
    importpath = "gopkg.in/src-d/go-billy.v3",
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
    srcs = [
        ":package-srcs",
        "//vendor/gopkg.in/src-d/go-billy.v3/helper/chroot:all-srcs",
        "//vendor/gopkg.in/src-d/go-billy.v3/helper/polyfill:all-srcs",
        "//vendor/gopkg.in/src-d/go-billy.v3/osfs:all-srcs",
        "//vendor/gopkg.in/src-d/go-billy.v3/util:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
