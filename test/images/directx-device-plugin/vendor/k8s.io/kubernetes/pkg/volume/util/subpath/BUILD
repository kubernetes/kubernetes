load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "subpath.go",
        "subpath_linux.go",
        "subpath_nsenter.go",
        "subpath_unsupported.go",
        "subpath_windows.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/volume/util/subpath",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//pkg/util/mount:go_default_library",
            "//vendor/k8s.io/utils/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//pkg/util/mount:go_default_library",
            "//vendor/k8s.io/utils/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//pkg/util/mount:go_default_library",
            "//vendor/k8s.io/utils/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//pkg/util/mount:go_default_library",
            "//vendor/k8s.io/utils/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//pkg/util/mount:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/k8s.io/klog:go_default_library",
            "//vendor/k8s.io/utils/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//pkg/util/mount:go_default_library",
            "//vendor/k8s.io/utils/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//pkg/util/mount:go_default_library",
            "//vendor/k8s.io/utils/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//pkg/util/mount:go_default_library",
            "//vendor/k8s.io/utils/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//pkg/util/mount:go_default_library",
            "//vendor/k8s.io/utils/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//pkg/util/mount:go_default_library",
            "//vendor/k8s.io/utils/nsenter:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//pkg/util/mount:go_default_library",
            "//vendor/k8s.io/klog:go_default_library",
            "//vendor/k8s.io/utils/nsenter:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = [
        "subpath_linux_test.go",
        "subpath_nsenter_test.go",
        "subpath_windows_test.go",
    ],
    embed = [":go_default_library"],
    deps = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//pkg/util/mount:go_default_library",
            "//pkg/volume/util/nsenter:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/k8s.io/klog:go_default_library",
            "//vendor/k8s.io/utils/nsenter:go_default_library",
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
