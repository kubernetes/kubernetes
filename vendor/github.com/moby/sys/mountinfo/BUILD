load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "mountinfo.go",
        "mountinfo_filters.go",
        "mountinfo_freebsd.go",
        "mountinfo_linux.go",
        "mountinfo_unsupported.go",
        "mountinfo_windows.go",
    ],
    cgo = True,
    importmap = "k8s.io/kubernetes/vendor/github.com/moby/sys/mountinfo",
    importpath = "github.com/moby/sys/mountinfo",
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
