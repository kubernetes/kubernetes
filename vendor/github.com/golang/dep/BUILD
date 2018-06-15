load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "analyzer.go",
        "context.go",
        "doc.go",
        "lock.go",
        "manifest.go",
        "project.go",
        "txn_writer.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/golang/dep",
    importpath = "github.com/golang/dep",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/golang/dep/gps:go_default_library",
        "//vendor/github.com/golang/dep/gps/paths:go_default_library",
        "//vendor/github.com/golang/dep/gps/pkgtree:go_default_library",
        "//vendor/github.com/golang/dep/internal/fs:go_default_library",
        "//vendor/github.com/pelletier/go-toml:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/golang/dep/cmd/kdep:all-srcs",
        "//vendor/github.com/golang/dep/gps:all-srcs",
        "//vendor/github.com/golang/dep/internal/dependencies:all-srcs",
        "//vendor/github.com/golang/dep/internal/feedback:all-srcs",
        "//vendor/github.com/golang/dep/internal/fs:all-srcs",
        "//vendor/github.com/golang/dep/internal/importers:all-srcs",
        "//vendor/github.com/golang/dep/internal/kdep:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
