load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "idtools.go",
        "idtools_unix.go",
        "idtools_windows.go",
        "usergroupadd_linux.go",
        "usergroupadd_unsupported.go",
        "utils_unix.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/pkg/idtools",
    importpath = "github.com/docker/docker/pkg/idtools",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/github.com/docker/docker/pkg/system:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/github.com/docker/docker/pkg/system:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/github.com/docker/docker/pkg/system:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/github.com/docker/docker/pkg/system:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/docker/docker/pkg/system:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//vendor/github.com/docker/docker/pkg/system:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/github.com/docker/docker/pkg/system:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//vendor/github.com/docker/docker/pkg/system:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//vendor/github.com/docker/docker/pkg/system:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/github.com/docker/docker/pkg/system:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/docker/docker/pkg/system:go_default_library",
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
