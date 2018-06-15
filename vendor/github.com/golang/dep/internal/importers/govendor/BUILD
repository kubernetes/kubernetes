load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["importer.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/golang/dep/internal/importers/govendor",
    importpath = "github.com/golang/dep/internal/importers/govendor",
    visibility = ["//vendor/github.com/golang/dep:__subpackages__"],
    deps = [
        "//vendor/github.com/golang/dep:go_default_library",
        "//vendor/github.com/golang/dep/gps:go_default_library",
        "//vendor/github.com/golang/dep/internal/importers/base:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
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
