load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["loader.go"],
    importmap = "k8s.io/kubernetes/vendor/honnef.co/go/tools/loader",
    importpath = "honnef.co/go/tools/loader",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/tools/go/gcexportdata:go_default_library",
        "//vendor/golang.org/x/tools/go/packages:go_default_library",
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
