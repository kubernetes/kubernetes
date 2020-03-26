load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["dlopen.go"],
    cgo = True,
    clinkopts = ["-ldl"],
    importmap = "k8s.io/kubernetes/vendor/github.com/coreos/go-systemd/v22/internal/dlopen",
    importpath = "github.com/coreos/go-systemd/v22/internal/dlopen",
    visibility = ["//vendor/github.com/coreos/go-systemd/v22:__subpackages__"],
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
