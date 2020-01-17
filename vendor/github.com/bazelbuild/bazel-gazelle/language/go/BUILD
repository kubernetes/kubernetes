load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "config.go",
        "constants.go",
        "dep.go",
        "fileinfo.go",
        "fix.go",
        "generate.go",
        "godep.go",
        "kinds.go",
        "known_go_imports.go",
        "known_proto_imports.go",
        "lang.go",
        "modules.go",
        "package.go",
        "resolve.go",
        "std_package_list.go",
        "update.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/bazelbuild/bazel-gazelle/language/go",
    importpath = "github.com/bazelbuild/bazel-gazelle/language/go",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/bazelbuild/bazel-gazelle/config:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/flag:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/label:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/language:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/language/proto:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/pathtools:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/repo:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/resolve:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/rule:go_default_library",
        "//vendor/github.com/bazelbuild/buildtools/build:go_default_library",
        "//vendor/github.com/pelletier/go-toml:go_default_library",
        "//vendor/golang.org/x/sync/errgroup:go_default_library",
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
