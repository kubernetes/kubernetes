load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "ansi_reader.go",
        "ansi_writer.go",
        "console.go",
        "windows.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/moby/term/windows",
    importpath = "github.com/moby/term/windows",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/Azure/go-ansiterm:go_default_library",
            "//vendor/github.com/Azure/go-ansiterm/winterm:go_default_library",
            "//vendor/github.com/sirupsen/logrus:go_default_library",
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
