load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "poly1305.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:amd64": [
            "sum_amd64.go",
            "sum_amd64.s",
        ],
        "//conditions:default": [],
    }) + select({
        "@io_bazel_rules_go//go/platform:android_386": [
            "sum_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:android_arm": [
            "sum_arm.go",
            "sum_arm.s",
        ],
        "@io_bazel_rules_go//go/platform:android_arm64": [
            "sum_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin_386": [
            "sum_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin_arm": [
            "sum_arm.go",
            "sum_arm.s",
        ],
        "@io_bazel_rules_go//go/platform:darwin_arm64": [
            "sum_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd_386": [
            "sum_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd_arm": [
            "sum_arm.go",
            "sum_arm.s",
        ],
        "@io_bazel_rules_go//go/platform:linux_386": [
            "sum_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_arm": [
            "sum_arm.go",
            "sum_arm.s",
        ],
        "@io_bazel_rules_go//go/platform:linux_arm64": [
            "sum_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_mips": [
            "sum_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_mips64": [
            "sum_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_mips64le": [
            "sum_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_mipsle": [
            "sum_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_ppc64": [
            "sum_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_ppc64le": [
            "sum_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_s390x": [
            "sum_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl_386": [
            "sum_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl_amd64p32": [
            "sum_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl_arm": [
            "sum_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd_386": [
            "sum_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd_arm": [
            "sum_arm.go",
            "sum_arm.s",
        ],
        "@io_bazel_rules_go//go/platform:openbsd_386": [
            "sum_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd_arm": [
            "sum_arm.go",
            "sum_arm.s",
        ],
        "@io_bazel_rules_go//go/platform:plan9_386": [
            "sum_ref.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9_arm": [
            "sum_arm.go",
            "sum_arm.s",
        ],
        "@io_bazel_rules_go//go/platform:windows_386": [
            "sum_ref.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "golang.org/x/crypto/poly1305",
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
