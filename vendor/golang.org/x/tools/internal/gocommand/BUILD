load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "invoke.go",
        "vendor.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/tools/internal/gocommand",
    importpath = "golang.org/x/tools/internal/gocommand",
    visibility = ["//vendor/golang.org/x/tools:__subpackages__"],
    deps = [
        "//vendor/golang.org/x/mod/semver:go_default_library",
        "//vendor/golang.org/x/tools/internal/event:go_default_library",
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
