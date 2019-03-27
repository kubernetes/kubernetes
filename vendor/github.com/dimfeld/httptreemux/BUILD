load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "context.go",
        "group.go",
        "panichandler.go",
        "path.go",
        "router.go",
        "tree.go",
        "treemux_16.go",
        "treemux_17.go",
        "unescape_17.go",
        "unescape_18.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/dimfeld/httptreemux",
    importpath = "github.com/dimfeld/httptreemux",
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "context_test.go",
        "fallthrough_test.go",
        "group_test.go",
        "path_test.go",
        "router_test.go",
        "tree_test.go",
    ],
    embed = [":go_default_library"],
    tags = ["manual"],
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
