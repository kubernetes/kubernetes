load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "init_shared.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "init_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "init_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "init_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "init_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "init_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "init_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "init_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "init_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "init_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "init_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "init_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/cmd/platform",
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
