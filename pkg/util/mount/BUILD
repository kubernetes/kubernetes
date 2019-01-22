load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "exec.go",
        "exec_mount.go",
        "exec_mount_unsupported.go",
        "fake.go",
        "mount.go",
        "mount_helper.go",
        "mount_linux.go",
        "mount_unsupported.go",
        "mount_windows.go",
        "nsenter_mount.go",
        "nsenter_mount_unsupported.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/util/mount",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/k8s.io/klog:go_default_library",
        "//vendor/k8s.io/utils/exec:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//pkg/util/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//pkg/util/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//pkg/util/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//pkg/util/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//pkg/util/file:go_default_library",
            "//pkg/util/io:go_default_library",
            "//pkg/util/nsenter:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//pkg/util/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//pkg/util/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//pkg/util/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//pkg/util/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//pkg/util/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//pkg/util/file:go_default_library",
            "//pkg/util/nsenter:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = [
        "exec_mount_test.go",
        "mount_helper_test.go",
        "mount_linux_test.go",
        "mount_test.go",
        "mount_windows_test.go",
        "nsenter_mount_test.go",
        "safe_format_and_mount_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//vendor/k8s.io/utils/exec/testing:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//pkg/util/nsenter:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/k8s.io/klog:go_default_library",
            "//vendor/k8s.io/utils/exec:go_default_library",
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
