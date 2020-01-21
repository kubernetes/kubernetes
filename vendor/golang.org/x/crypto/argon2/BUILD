load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "argon2.go",
        "blake2b.go",
        "blamka_amd64.go",
        "blamka_amd64.s",
        "blamka_generic.go",
        "blamka_ref.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/crypto/argon2",
    importpath = "golang.org/x/crypto/argon2",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/crypto/blake2b:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:amd64": [
            "//vendor/golang.org/x/sys/cpu:go_default_library",
        ],
        "//conditions:default": [],
    }),
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
