load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "subpath.go",
        "subpath_linux.go",
        "subpath_unsupported.go",
        "subpath_windows.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/volume/util/subpath",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:aix": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
            "//vendor/k8s.io/utils/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:android": [
            "//pkg/volume/util/hostutil:go_default_library",
            "//staging/src/k8s.io/mount-utils:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
            "//vendor/k8s.io/utils/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
            "//vendor/k8s.io/utils/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
            "//vendor/k8s.io/utils/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:illumos": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
            "//vendor/k8s.io/utils/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:ios": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
            "//vendor/k8s.io/utils/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:js": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
            "//vendor/k8s.io/utils/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//pkg/volume/util/hostutil:go_default_library",
            "//staging/src/k8s.io/mount-utils:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
            "//vendor/k8s.io/utils/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
            "//vendor/k8s.io/utils/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
            "//vendor/k8s.io/utils/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
            "//vendor/k8s.io/utils/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
            "//vendor/k8s.io/utils/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
            "//vendor/k8s.io/utils/nsenter:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = [
        "subpath_linux_test.go",
        "subpath_windows_test.go",
    ],
    embed = [":go_default_library"],
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//staging/src/k8s.io/mount-utils:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/stretchr/testify/assert:go_default_library",
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
