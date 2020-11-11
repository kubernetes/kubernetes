load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "licenses.go",
        "main.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.uber.org/tools/update-license",
    importpath = "go.uber.org/tools/update-license",
    visibility = ["//visibility:private"],
)

go_binary(
    name = "update-license",
    embed = [":go_default_library"],
    tags = ["manual"],
    visibility = ["//visibility:public"],
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
