load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "volume.go",
        "volume_driver_registry.go",
        "volume_not_supported.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/libopenstorage/openstorage/volume",
    importpath = "github.com/libopenstorage/openstorage/volume",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/libopenstorage/openstorage/api:go_default_library"],
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
