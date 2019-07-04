load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "util.go",
        "util_cgo.go",
        "util_stub.go",
    ],
    cgo = True,
    importmap = "k8s.io/kubernetes/vendor/github.com/coreos/go-systemd/util",
    importpath = "github.com/coreos/go-systemd/util",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/coreos/pkg/dlopen:go_default_library"],
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
