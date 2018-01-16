load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = select({
        "@io_bazel_rules_go//go/platform:android": [
            "nsenter_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "nsenter_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "nsenter_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "nsenter_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "nsenter.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "nsenter_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "nsenter_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "nsenter_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "nsenter_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "nsenter_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "nsenter_unsupported.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/util/nsenter",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/k8s.io/utils/exec:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/k8s.io/utils/exec:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/k8s.io/utils/exec:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/k8s.io/utils/exec:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/k8s.io/utils/exec:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//vendor/k8s.io/utils/exec:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/k8s.io/utils/exec:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//vendor/k8s.io/utils/exec:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//vendor/k8s.io/utils/exec:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/k8s.io/utils/exec:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
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
