load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "resizefs_linux.go",
        "resizefs_unsupported.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/util/resizefs",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:aix": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:android": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:illumos": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:ios": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:js": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
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
