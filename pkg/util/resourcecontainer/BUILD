package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = select({
        "@io_bazel_rules_go//go/platform:android": [
            "resource_container_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "resource_container_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "resource_container_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "resource_container_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "resource_container_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "resource_container_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "resource_container_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "resource_container_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "resource_container_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "resource_container_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "resource_container_unsupported.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/util/resourcecontainer",
    deps = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/fs:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
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
)
