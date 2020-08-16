load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["module.go"],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/mod/module",
    importpath = "golang.org/x/mod/module",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/mod/semver:go_default_library",
        "//vendor/golang.org/x/xerrors:go_default_library",
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
