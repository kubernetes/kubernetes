package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["httpstream_test.go"],
    embed = [":go_default_library"],
    importpath = "k8s.io/apimachinery/pkg/util/httpstream",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "httpstream.go",
    ],
    importpath = "k8s.io/apimachinery/pkg/util/httpstream",
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
)
