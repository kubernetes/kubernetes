load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "ginkgo_ast_nodes.go",
        "import.go",
        "package_rewriter.go",
        "test_finder.go",
        "testfile_rewriter.go",
        "testing_t_rewriter.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/onsi/ginkgo/ginkgo/convert",
    importpath = "github.com/onsi/ginkgo/ginkgo/convert",
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
