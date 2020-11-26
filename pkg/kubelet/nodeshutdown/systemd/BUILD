load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "inhibit_linux.go",
        "inhibit_others.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/nodeshutdown/systemd",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/github.com/godbus/dbus/v5:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/godbus/dbus/v5:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = ["inhibit_linux_test.go"],
    embed = [":go_default_library"],
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/github.com/godbus/dbus/v5:go_default_library",
            "//vendor/github.com/stretchr/testify/assert:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/godbus/dbus/v5:go_default_library",
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
