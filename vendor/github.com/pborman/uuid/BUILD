load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "dce.go",
        "doc.go",
        "hash.go",
        "marshal.go",
        "node.go",
        "sql.go",
        "time.go",
        "util.go",
        "uuid.go",
        "version1.go",
        "version4.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/pborman/uuid",
    importpath = "github.com/pborman/uuid",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/google/uuid:go_default_library"],
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
