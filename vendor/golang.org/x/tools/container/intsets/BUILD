load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "sparse.go",
        "util.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:386": [
            "popcnt_generic.go",
        ],
        "@io_bazel_rules_go//go/platform:amd64": [
            "popcnt_amd64.go",
            "popcnt_amd64.s",
        ],
        "@io_bazel_rules_go//go/platform:amd64p32": [
            "popcnt_generic.go",
        ],
        "@io_bazel_rules_go//go/platform:arm": [
            "popcnt_generic.go",
        ],
        "@io_bazel_rules_go//go/platform:arm64": [
            "popcnt_generic.go",
        ],
        "@io_bazel_rules_go//go/platform:mips": [
            "popcnt_generic.go",
        ],
        "@io_bazel_rules_go//go/platform:mips64": [
            "popcnt_generic.go",
        ],
        "@io_bazel_rules_go//go/platform:mips64le": [
            "popcnt_generic.go",
        ],
        "@io_bazel_rules_go//go/platform:mipsle": [
            "popcnt_generic.go",
        ],
        "@io_bazel_rules_go//go/platform:ppc64": [
            "popcnt_generic.go",
        ],
        "@io_bazel_rules_go//go/platform:ppc64le": [
            "popcnt_generic.go",
        ],
        "@io_bazel_rules_go//go/platform:s390x": [
            "popcnt_generic.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "golang.org/x/tools/container/intsets",
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
