load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "ensure.go",
        "failures.go",
        "gopath_scanner.go",
        "graphviz.go",
        "hash_in.go",
        "init.go",
        "main.go",
        "prune.go",
        "root_analyzer.go",
        "status.go",
        "version.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/golang/dep/cmd/kdep",
    importpath = "github.com/golang/dep/cmd/kdep",
    visibility = ["//visibility:private"],
    deps = [
        "//vendor/github.com/golang/dep:go_default_library",
        "//vendor/github.com/golang/dep/gps:go_default_library",
        "//vendor/github.com/golang/dep/gps/paths:go_default_library",
        "//vendor/github.com/golang/dep/gps/pkgtree:go_default_library",
        "//vendor/github.com/golang/dep/internal/feedback:go_default_library",
        "//vendor/github.com/golang/dep/internal/fs:go_default_library",
        "//vendor/github.com/golang/dep/internal/importers:go_default_library",
        "//vendor/github.com/golang/dep/internal/kdep:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/golang.org/x/sync/errgroup:go_default_library",
    ],
)

go_binary(
    name = "kdep",
    embed = [":go_default_library"],
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
