load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "osext.go",
        "osext_plan9.go",
        "osext_procfs.go",
        "osext_sysctl.go",
        "osext_windows.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/kardianos/osext",
    importpath = "github.com/kardianos/osext",
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
