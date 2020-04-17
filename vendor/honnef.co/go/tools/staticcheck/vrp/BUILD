load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "channel.go",
        "int.go",
        "slice.go",
        "string.go",
        "vrp.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/honnef.co/go/tools/staticcheck/vrp",
    importpath = "honnef.co/go/tools/staticcheck/vrp",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/honnef.co/go/tools/lint:go_default_library",
        "//vendor/honnef.co/go/tools/ssa:go_default_library",
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
