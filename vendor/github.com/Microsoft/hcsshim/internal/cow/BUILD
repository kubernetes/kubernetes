load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["cow.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/Microsoft/hcsshim/internal/cow",
    importpath = "github.com/Microsoft/hcsshim/internal/cow",
    visibility = ["//vendor/github.com/Microsoft/hcsshim:__subpackages__"],
    deps = [
        "//vendor/github.com/Microsoft/hcsshim/internal/schema1:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/schema2:go_default_library",
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
