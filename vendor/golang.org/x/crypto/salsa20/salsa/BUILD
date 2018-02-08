load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "hsalsa20.go",
        "salsa208.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:386": [
            "salsa20_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:amd64": [
            "salsa2020_amd64.s",
            "salsa20_amd64.go",
        ],
        "@io_bazel_rules_go//go/platform:amd64p32": [
            "salsa20_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:arm": [
            "salsa20_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:arm64": [
            "salsa20_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:mips": [
            "salsa20_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:mips64": [
            "salsa20_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:mips64le": [
            "salsa20_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:mipsle": [
            "salsa20_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:ppc64": [
            "salsa20_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:ppc64le": [
            "salsa20_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:s390x": [
            "salsa20_ref.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "golang.org/x/crypto/salsa20/salsa",
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
