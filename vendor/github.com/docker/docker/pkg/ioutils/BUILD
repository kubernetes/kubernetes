load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "buffer.go",
        "bytespipe.go",
        "fswriters.go",
        "readers.go",
        "temp_unix.go",
        "temp_windows.go",
        "writeflusher.go",
        "writers.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/pkg/ioutils",
    importpath = "github.com/docker/docker/pkg/ioutils",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/net/context:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/docker/docker/pkg/longpath:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
