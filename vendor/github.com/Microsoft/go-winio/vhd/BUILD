load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "vhd.go",
        "zvhd.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/Microsoft/go-winio/vhd",
    importpath = "github.com/Microsoft/go-winio/vhd",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/sys/windows:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/Microsoft/go-winio/pkg/guid:go_default_library",
            "//vendor/github.com/pkg/errors:go_default_library",
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
