load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "chacha_arm64.go",
        "chacha_arm64.s",
        "chacha_generic.go",
        "chacha_noasm.go",
        "chacha_ppc64le.go",
        "chacha_ppc64le.s",
        "chacha_s390x.go",
        "chacha_s390x.s",
        "xor.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/crypto/chacha20",
    importpath = "golang.org/x/crypto/chacha20",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/crypto/internal/subtle:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:s390x": [
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
