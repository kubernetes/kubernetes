load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "fake_hostutil.go",
        "hostutil.go",
        "hostutil_linux.go",
        "hostutil_unsupported.go",
        "hostutil_windows.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/volume/util/hostutil",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/k8s.io/utils/mount:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
            "//vendor/k8s.io/utils/path:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
            "//vendor/k8s.io/utils/path:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/k8s.io/klog/v2:go_default_library",
            "//vendor/k8s.io/utils/path:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = [
        "hostutil_linux_test.go",
        "hostutil_windows_test.go",
    ],
    embed = [":go_default_library"],
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/k8s.io/utils/exec:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/k8s.io/utils/exec:go_default_library",
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
