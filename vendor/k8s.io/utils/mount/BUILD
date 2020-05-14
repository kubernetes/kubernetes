load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "fake_mounter.go",
        "mount.go",
        "mount_helper_common.go",
        "mount_helper_unix.go",
        "mount_helper_windows.go",
        "mount_linux.go",
        "mount_unsupported.go",
        "mount_windows.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/utils/mount",
    importpath = "k8s.io/utils/mount",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/k8s.io/klog/v2:go_default_library",
        "//vendor/k8s.io/utils/exec:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/k8s.io/utils/io:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/k8s.io/utils/io:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/k8s.io/utils/io:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/k8s.io/utils/io:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:ios": [
            "//vendor/k8s.io/utils/io:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/k8s.io/utils/io:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//vendor/k8s.io/utils/io:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/k8s.io/utils/io:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//vendor/k8s.io/utils/io:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//vendor/k8s.io/utils/io:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/k8s.io/utils/io:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/k8s.io/utils/keymutex:go_default_library",
            "//vendor/k8s.io/utils/path:go_default_library",
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
