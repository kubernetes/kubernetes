package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "cgo_genrule",
    "go_library",
)

cgo_genrule(
    name = "cgo_codegen",
    srcs = [
        "dlopen.go",
        "dlopen_example.go",
    ],
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
        "dlopen.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "dlopen_example.go",
        ],
        "//conditions:default": [],
    }),
    cgo = True,
    clinkopts = ["-ldl"],
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
