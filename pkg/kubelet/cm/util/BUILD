package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = select({
        "@io_bazel_rules_go//go/platform:android": [
            "cgroups_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "cgroups_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "cgroups_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "cgroups_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "cgroups_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "cgroups_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "cgroups_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "cgroups_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "cgroups_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "cgroups_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "cgroups_unsupported.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/kubelet/cm/util",
    deps = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/utils:go_default_library",
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
