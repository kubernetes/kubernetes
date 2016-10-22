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
        "resize.go",
        "resizeevents.go",
        "setsize.go",
        "term.go",
        "term_writer.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/util/interrupt:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//vendor:github.com/docker/docker/pkg/term",
        "//vendor:github.com/mitchellh/go-wordwrap",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["term_writer_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
