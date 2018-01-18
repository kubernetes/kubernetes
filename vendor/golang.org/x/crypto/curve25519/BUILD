load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:386": [
            "curve25519.go",
        ],
        "@io_bazel_rules_go//go/platform:amd64": [
            "const_amd64.h",
            "const_amd64.s",
            "cswap_amd64.s",
            "freeze_amd64.s",
            "ladderstep_amd64.s",
            "mont25519_amd64.go",
            "mul_amd64.s",
            "square_amd64.s",
        ],
        "@io_bazel_rules_go//go/platform:amd64p32": [
            "curve25519.go",
        ],
        "@io_bazel_rules_go//go/platform:arm": [
            "curve25519.go",
        ],
        "@io_bazel_rules_go//go/platform:arm64": [
            "curve25519.go",
        ],
        "@io_bazel_rules_go//go/platform:mips": [
            "curve25519.go",
        ],
        "@io_bazel_rules_go//go/platform:mips64": [
            "curve25519.go",
        ],
        "@io_bazel_rules_go//go/platform:mips64le": [
            "curve25519.go",
        ],
        "@io_bazel_rules_go//go/platform:mipsle": [
            "curve25519.go",
        ],
        "@io_bazel_rules_go//go/platform:ppc64": [
            "curve25519.go",
        ],
        "@io_bazel_rules_go//go/platform:ppc64le": [
            "curve25519.go",
        ],
        "@io_bazel_rules_go//go/platform:s390x": [
            "curve25519.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "golang.org/x/crypto/curve25519",
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
