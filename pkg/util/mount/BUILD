load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "exec.go",
        "fake.go",
        "mount.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "exec_mount_unsupported.go",
            "mount_unsupported.go",
            "nsenter_mount_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "exec_mount_unsupported.go",
            "mount_unsupported.go",
            "nsenter_mount_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "exec_mount_unsupported.go",
            "mount_unsupported.go",
            "nsenter_mount_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "exec_mount_unsupported.go",
            "mount_unsupported.go",
            "nsenter_mount_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "exec_mount.go",
            "mount_linux.go",
            "nsenter_mount.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "exec_mount_unsupported.go",
            "mount_unsupported.go",
            "nsenter_mount_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "exec_mount_unsupported.go",
            "mount_unsupported.go",
            "nsenter_mount_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "exec_mount_unsupported.go",
            "mount_unsupported.go",
            "nsenter_mount_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "exec_mount_unsupported.go",
            "mount_unsupported.go",
            "nsenter_mount_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "exec_mount_unsupported.go",
            "mount_unsupported.go",
            "nsenter_mount_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "exec_mount_unsupported.go",
            "mount_windows.go",
            "nsenter_mount_unsupported.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/util/mount",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/utils/exec:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//pkg/util/io:go_default_library",
            "//pkg/util/nsenter:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = [
        "safe_format_and_mount_test.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
            "exec_mount_test.go",
            "mount_linux_test.go",
            "nsenter_mount_test.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "mount_windows_test.go",
        ],
        "//conditions:default": [],
    }),
    embed = [":go_default_library"],
    deps = [
        "//vendor/k8s.io/utils/exec/testing:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/golang/glog:go_default_library",
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
