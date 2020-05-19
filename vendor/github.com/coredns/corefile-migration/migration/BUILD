load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "migrate.go",
        "notice.go",
        "plugins.go",
        "versions.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/coredns/corefile-migration/migration",
    importpath = "github.com/coredns/corefile-migration/migration",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/coredns/corefile-migration/migration/corefile:go_default_library"],
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
        "//vendor/github.com/coredns/corefile-migration/migration/corefile:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
