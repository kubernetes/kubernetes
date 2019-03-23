load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "annotations.go",
        "config.go",
        "descriptor.go",
        "index.go",
        "layout.go",
        "manifest.go",
        "mediatype.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/opencontainers/image-spec/specs-go/v1",
    importpath = "github.com/opencontainers/image-spec/specs-go/v1",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/opencontainers/go-digest:go_default_library",
        "//vendor/github.com/opencontainers/image-spec/specs-go:go_default_library",
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
