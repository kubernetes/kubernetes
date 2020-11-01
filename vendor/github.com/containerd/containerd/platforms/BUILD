load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "compare.go",
        "cpuinfo.go",
        "database.go",
        "defaults.go",
        "defaults_unix.go",
        "defaults_windows.go",
        "platforms.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/containerd/containerd/platforms",
    importpath = "github.com/containerd/containerd/platforms",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/containerd/containerd/errdefs:go_default_library",
        "//vendor/github.com/containerd/containerd/log:go_default_library",
        "//vendor/github.com/opencontainers/image-spec/specs-go/v1:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
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
