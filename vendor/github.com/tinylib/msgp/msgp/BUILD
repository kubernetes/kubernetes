package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "advise_linux.go",
        "circular.go",
        "defs.go",
        "edit.go",
        "elsize.go",
        "errors.go",
        "extension.go",
        "file.go",
        "integers.go",
        "json.go",
        "json_bytes.go",
        "number.go",
        "read.go",
        "read_bytes.go",
        "size.go",
        "unsafe.go",
        "write.go",
        "write_bytes.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor/github.com/philhofer/fwd:go_default_library"],
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
)
