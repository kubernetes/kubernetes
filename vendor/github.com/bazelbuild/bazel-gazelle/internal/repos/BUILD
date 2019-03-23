load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "dep.go",
        "modules.go",
        "remote.go",
        "repo.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/bazelbuild/bazel-gazelle/internal/repos",
    importpath = "github.com/bazelbuild/bazel-gazelle/internal/repos",
    visibility = ["//vendor/github.com/bazelbuild/bazel-gazelle:__subpackages__"],
    deps = [
        "//vendor/github.com/bazelbuild/bazel-gazelle/internal/label:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/internal/pathtools:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/internal/rule:go_default_library",
        "//vendor/github.com/pelletier/go-toml:go_default_library",
        "//vendor/golang.org/x/tools/go/vcs:go_default_library",
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
