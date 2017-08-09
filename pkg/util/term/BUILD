package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "setsize.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "setsize_unsupported.go",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//vendor/github.com/docker/docker/pkg/term:go_default_library",
        "//vendor/k8s.io/client-go/tools/remotecommand:go_default_library",
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
)
