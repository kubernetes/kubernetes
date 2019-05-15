load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "pidlimit.go",
        "pidlimit_linux.go",
        "pidlimit_unsupported.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/kubelet/pkg/stats/pidlimit",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//cmd/kubelet/pkg/apis/stats/v1alpha1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//cmd/kubelet/pkg/apis/stats/v1alpha1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//cmd/kubelet/pkg/apis/stats/v1alpha1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//cmd/kubelet/pkg/apis/stats/v1alpha1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//cmd/kubelet/pkg/apis/stats/v1alpha1:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//cmd/kubelet/pkg/apis/stats/v1alpha1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//cmd/kubelet/pkg/apis/stats/v1alpha1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//cmd/kubelet/pkg/apis/stats/v1alpha1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//cmd/kubelet/pkg/apis/stats/v1alpha1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//cmd/kubelet/pkg/apis/stats/v1alpha1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//cmd/kubelet/pkg/apis/stats/v1alpha1:go_default_library",
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
