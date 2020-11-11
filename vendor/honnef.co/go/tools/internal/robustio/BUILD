load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "robustio.go",
        "robustio_darwin.go",
        "robustio_flaky.go",
        "robustio_other.go",
        "robustio_windows.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/honnef.co/go/tools/internal/robustio",
    importpath = "honnef.co/go/tools/internal/robustio",
    visibility = ["//vendor/honnef.co/go/tools:__subpackages__"],
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
