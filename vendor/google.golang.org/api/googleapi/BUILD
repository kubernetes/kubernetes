load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "googleapi.go",
        "types.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/api/googleapi",
    importpath = "google.golang.org/api/googleapi",
    visibility = ["//visibility:public"],
    deps = ["//vendor/google.golang.org/api/googleapi/internal/uritemplates:go_default_library"],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//vendor/google.golang.org/api/googleapi/internal/uritemplates:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
