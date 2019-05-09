load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "colorable_others.go",
        "colorable_windows.go",
        "noncolorable.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/onsi/ginkgo/reporters/stenographer/support/go-colorable",
    importpath = "github.com/onsi/ginkgo/reporters/stenographer/support/go-colorable",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/onsi/ginkgo/reporters/stenographer/support/go-isatty:go_default_library",
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
