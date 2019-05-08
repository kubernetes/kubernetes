load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "exec.go",
        "fake.go",
        "mount.go",
        "mount_helper_common.go",
        "mount_helper_unix.go",
        "mount_helper_windows.go",
        "mount_linux.go",
        "mount_unsupported.go",
        "mount_windows.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/util/mount",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/k8s.io/klog:go_default_library",
        "//vendor/k8s.io/utils/exec:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/k8s.io/utils/io:go_default_library",
            "//vendor/k8s.io/utils/path:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/k8s.io/utils/keymutex:go_default_library",
            "//vendor/k8s.io/utils/path:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = [
        "mount_helper_test.go",
        "mount_linux_test.go",
        "mount_test.go",
        "mount_windows_test.go",
        "safe_format_and_mount_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//vendor/k8s.io/utils/exec/testing:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
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
