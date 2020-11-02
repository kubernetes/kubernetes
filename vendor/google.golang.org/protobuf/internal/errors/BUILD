load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "errors.go",
        "is_go112.go",
        "is_go113.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/protobuf/internal/errors",
    importpath = "google.golang.org/protobuf/internal/errors",
    visibility = ["//vendor/google.golang.org/protobuf:__subpackages__"],
    deps = ["//vendor/google.golang.org/protobuf/internal/detrand:go_default_library"],
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
