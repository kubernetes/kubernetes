load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["spec_handler.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/libopenstorage/openstorage/api/spec",
    importpath = "github.com/libopenstorage/openstorage/api/spec",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/libopenstorage/openstorage/api:go_default_library",
        "//vendor/github.com/libopenstorage/openstorage/pkg/parser:go_default_library",
        "//vendor/github.com/libopenstorage/openstorage/pkg/units:go_default_library",
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
