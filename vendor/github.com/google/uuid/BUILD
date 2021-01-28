load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "dce.go",
        "doc.go",
        "hash.go",
        "marshal.go",
        "node.go",
        "node_js.go",
        "node_net.go",
        "sql.go",
        "time.go",
        "util.go",
        "uuid.go",
        "version1.go",
        "version4.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/google/uuid",
    importpath = "github.com/google/uuid",
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
