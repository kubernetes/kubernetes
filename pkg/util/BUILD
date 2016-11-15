package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "template.go",
        "trace.go",
        "trie.go",
        "umask.go",
        "util.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/glog"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "template_test.go",
        "util_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/util/diff:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
    ],
)
