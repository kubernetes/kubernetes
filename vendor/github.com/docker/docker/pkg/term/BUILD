package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "cgo_genrule",
    "go_library",
)

cgo_genrule(
    name = "cgo_codegen",
    srcs = ["tc_linux_cgo.go"],
    clinkopts = [
        "-lz",
        "-lm",
        "-lpthread",
        "-ldl",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "go_default_library",
    srcs = [
        "ascii.go",
        "term.go",
        "term_unix.go",
    ],
    library = ":cgo_codegen",
    tags = ["automanaged"],
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
        "//vendor/github.com/docker/docker/pkg/term/windows:all-srcs",
    ],
    tags = ["automanaged"],
)
