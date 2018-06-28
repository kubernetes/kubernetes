load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "context.go",
        "kdep.go",
        "manifest.go",
        "project.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/golang/dep/internal/kdep",
    importpath = "github.com/golang/dep/internal/kdep",
    visibility = ["//vendor/github.com/golang/dep:__subpackages__"],
    deps = [
        "//vendor/github.com/golang/dep:go_default_library",
        "//vendor/github.com/golang/dep/gps:go_default_library",
        "//vendor/github.com/golang/dep/gps/pkgtree:go_default_library",
        "//vendor/github.com/golang/dep/internal/dependencies:go_default_library",
        "//vendor/github.com/pelletier/go-toml:go_default_library",
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
