load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "buildozer.go",
        "edit.go",
        "fix.go",
        "types.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/bazelbuild/buildtools/edit",
    importpath = "github.com/bazelbuild/buildtools/edit",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/bazelbuild/buildtools/api_proto:go_default_library",
        "//vendor/github.com/bazelbuild/buildtools/build:go_default_library",
        "//vendor/github.com/bazelbuild/buildtools/build_proto:go_default_library",
        "//vendor/github.com/bazelbuild/buildtools/file:go_default_library",
        "//vendor/github.com/bazelbuild/buildtools/lang:go_default_library",
        "//vendor/github.com/bazelbuild/buildtools/tables:go_default_library",
        "//vendor/github.com/bazelbuild/buildtools/wspace:go_default_library",
        "//vendor/github.com/golang/protobuf/proto:go_default_library",
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
