load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["api_base.pb.go"],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/appengine/internal/base",
    importpath = "google.golang.org/appengine/internal/base",
    visibility = ["//vendor/google.golang.org/appengine:__subpackages__"],
    deps = ["//vendor/github.com/golang/protobuf/proto:go_default_library"],
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
