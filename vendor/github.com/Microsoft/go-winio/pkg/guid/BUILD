load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["guid.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/Microsoft/go-winio/pkg/guid",
    importpath = "github.com/Microsoft/go-winio/pkg/guid",
    visibility = ["//visibility:public"],
    deps = ["//vendor/golang.org/x/sys/windows:go_default_library"],
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
