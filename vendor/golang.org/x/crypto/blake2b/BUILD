load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "blake2b.go",
        "blake2bAVX2_amd64.go",
        "blake2bAVX2_amd64.s",
        "blake2b_amd64.go",
        "blake2b_amd64.s",
        "blake2b_generic.go",
        "blake2b_ref.go",
        "blake2x.go",
        "register.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/crypto/blake2b",
    importpath = "golang.org/x/crypto/blake2b",
    visibility = ["//visibility:public"],
    deps = select({
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
