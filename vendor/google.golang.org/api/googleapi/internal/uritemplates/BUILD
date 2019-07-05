load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "uritemplates.go",
        "utils.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/api/googleapi/internal/uritemplates",
    importpath = "google.golang.org/api/googleapi/internal/uritemplates",
    visibility = ["//vendor/google.golang.org/api/googleapi:__subpackages__"],
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
