load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "curve25519.go",
        "doc.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "const_amd64.h",
            "const_amd64.s",
            "cswap_amd64.s",
            "freeze_amd64.s",
            "ladderstep_amd64.s",
            "mont25519_amd64.go",
            "mul_amd64.s",
            "square_amd64.s",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "const_amd64.h",
            "const_amd64.s",
            "cswap_amd64.s",
            "freeze_amd64.s",
            "ladderstep_amd64.s",
            "mont25519_amd64.go",
            "mul_amd64.s",
            "square_amd64.s",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "const_amd64.h",
            "const_amd64.s",
            "cswap_amd64.s",
            "freeze_amd64.s",
            "ladderstep_amd64.s",
            "mont25519_amd64.go",
            "mul_amd64.s",
            "square_amd64.s",
        ],
        "//conditions:default": [],
    }),
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
