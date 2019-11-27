load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "context.go",
        "doc.go",
        "key.go",
        "map.go",
        "map_codec.go",
        "metadata.go",
        "profile_19.go",
        "profile_not19.go",
        "validate.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.opencensus.io/tag",
    importpath = "go.opencensus.io/tag",
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
