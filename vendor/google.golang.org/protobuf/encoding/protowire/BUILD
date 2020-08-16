load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["wire.go"],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/protobuf/encoding/protowire",
    importpath = "google.golang.org/protobuf/encoding/protowire",
    visibility = ["//visibility:public"],
    deps = ["//vendor/google.golang.org/protobuf/internal/errors:go_default_library"],
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
