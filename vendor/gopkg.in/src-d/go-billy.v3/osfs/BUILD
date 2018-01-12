load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "os.go",
        "os_posix.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "os_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "gopkg.in/src-d/go-billy.v3/osfs",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/gopkg.in/src-d/go-billy.v3:go_default_library",
        "//vendor/gopkg.in/src-d/go-billy.v3/helper/chroot:go_default_library",
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
