load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "selinux.go",
        "selinux_linux.go",
        "selinux_stub.go",
        "xattrs.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/opencontainers/selinux/go-selinux",
    importpath = "github.com/opencontainers/selinux/go-selinux",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/pkg/errors:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/github.com/opencontainers/selinux/pkg/pwalk:go_default_library",
            "//vendor/github.com/willf/bitset:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/opencontainers/selinux/pkg/pwalk:go_default_library",
            "//vendor/github.com/willf/bitset:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/opencontainers/selinux/go-selinux/label:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
