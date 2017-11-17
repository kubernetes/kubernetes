load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "httpstream.go",
    ],
    importpath = "k8s.io/apimachinery/pkg/util/httpstream",
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["httpstream_test.go"],
    importpath = "k8s.io/apimachinery/pkg/util/httpstream",
    library = ":go_default_library",
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
        "//staging/src/k8s.io/apimachinery/pkg/util/httpstream/spdy:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
