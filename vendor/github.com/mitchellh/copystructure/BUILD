load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "copier_time.go",
        "copystructure.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/mitchellh/copystructure",
    importpath = "github.com/mitchellh/copystructure",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/mitchellh/reflectwalk:go_default_library"],
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
