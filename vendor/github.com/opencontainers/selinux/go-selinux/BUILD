load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "selinux_linux.go",
        "xattrs.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/opencontainers/selinux/go-selinux",
    importpath = "github.com/opencontainers/selinux/go-selinux",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
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
