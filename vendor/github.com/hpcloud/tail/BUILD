load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "tail.go",
        "tail_posix.go",
        "tail_windows.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/hpcloud/tail",
    importpath = "github.com/hpcloud/tail",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/hpcloud/tail/ratelimiter:go_default_library",
        "//vendor/github.com/hpcloud/tail/util:go_default_library",
        "//vendor/github.com/hpcloud/tail/watch:go_default_library",
        "//vendor/gopkg.in/tomb.v1:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/hpcloud/tail/winfile:go_default_library",
        ],
        "//conditions:default": [],
    }),
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
        "//vendor/github.com/hpcloud/tail/ratelimiter:all-srcs",
        "//vendor/github.com/hpcloud/tail/util:all-srcs",
        "//vendor/github.com/hpcloud/tail/watch:all-srcs",
        "//vendor/github.com/hpcloud/tail/winfile:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
