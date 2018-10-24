load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "buffer.go",
        "cmux.go",
        "doc.go",
        "matchers.go",
        "patricia.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/soheilhy/cmux",
    importpath = "github.com/soheilhy/cmux",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/net/http2:go_default_library",
        "//vendor/golang.org/x/net/http2/hpack:go_default_library",
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
